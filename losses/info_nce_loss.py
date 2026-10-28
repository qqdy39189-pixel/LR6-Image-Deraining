import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class InfoNCELoss(nn.Module):
    """课程感知损失变体的 InfoNCE 形式。

    和 easy_contrastive_loss.py 的对应关系（只改损失形式，不改编码器）：
      - 编码器不变：仍是 VGG19 features 前 35 层(索引 0~34),冻结、eval、
        ImageNet 归一化、AdaptiveAvgPool2d((1, 1)) 压成 512 维向量。
      - 三路角色不变：锚点 = 去雨预测，正样本 = 无雨目标，负样本 = 有雨输入。
      - 改的只有计分方式：
          原版:loss = (1 - sim_pos) + sim_neg,是线性的“拉近 + 推远”。
          本版:loss = -log[ exp(sim_pos / tau) /
                              (exp(sim_pos / tau) + sum_j exp(sim_neg_j / tau)) ]
                是 InfoNCE 的 log-softmax 形式：把“预测属于正样本”
                写成一个在 {正样本 + 全部负样本} 上的分类概率，再取负对数。

    为什么负样本是一批，而不只用自己那张有雨图：
      经典 InfoNCE 的分母是对一个负样本集合求和，只用 1 个负样本时它
      退化成二分类，虽然形式上也成立，但体现不出“从很多候选里挑出
      正样本”这个本意。所以这里对 batch 里的每张有雨图都当负样本：
      对锚点 i,负样本集合 = {rainy_j | j = 0..B-1}，其中 j = i 的那张
      （同场景有雨图）是最难的负样本，其它场景的有雨图是较易的负样本，
      相似度低时它们对分母的贡献自然很小，不会喧宾夺主。
      batch_size = 1 时，本版自动退化为单负样本的 InfoNCE。

    温度系数 tau:
      经典取值 0.07(MoCo)到 0.1~0.5(SimCLR)之间。本版默认 0.07,
      和标准对比学习保持一致。tau 越小，相似度的小差距被放得越大，
      梯度也越大；如果训练中损失震荡明显，可把 tau 调大到 0.1 或 0.2。
      tau 由 main.py 的 --temperature 传入。
    """

    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        self.temperature = temperature

        # 与感知损失、Easy 对比损失相同的预训练模型、相同的中间层：
        # VGG19 的 features 前 35 层（索引 0~34，到最后一个卷积层为止）
        weights = VGG19_Weights.DEFAULT
        vgg = vgg19(weights=weights)
        self.features = vgg.features[:35]

        # 冻结 VGG19 参数，只当特征提取器，不参与训练
        # （这一点和原版完全一样：改的是损失形式，不是编码器）
        for param in self.features.parameters():
            param.requires_grad = False

        # 固定为评估模式
        self.features.eval()

        # 使用 ImageNet 的均值和标准差进行归一化（与感知损失保持一致）
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # 自适应平均池化：不管特征图多大，每个通道都压成一个数
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _to_vector(self, image):
        # 一张图 -> 一条单位化后的特征向量：
        # 先归一化，再过 VGG 取特征图，然后每个通道取全图平均，
        # 最后摊平成 (B, 512)，并做 L2 归一化。
        # 归一化后，向量点积就等于余弦相似度，这正是 InfoNCE 的标准做法。
        image = (image - self.mean) / self.std
        feat_map = self.features(image)      # (B, 512, H, W)
        pooled = self.pool(feat_map)         # (B, 512, 1, 1)
        vec = pooled.flatten(1)              # (B, 512)
        return F.normalize(vec, p=2, dim=1)  # (B, 512)，每行模长为 1

    def forward(self, input_image, generated, ground_truth):
        # 三张图走同一条流水线，各得到一条单位向量
        # 注意参数顺序和原版、和 main.py 的调用保持一致：
        # input_image = 有雨输入，generated = 去雨预测，ground_truth = 无雨目标
        vec_input = self._to_vector(input_image)   # 有雨图，负样本集合
        vec_gen = self._to_vector(generated)       # 去雨预测，锚点
        vec_gt = self._to_vector(ground_truth)     # 无雨目标，正样本

        # 正样本相似度：锚点 i 只和自己那张干净图配对，取对角线
        # sim_pos[i] = vec_gen[i] . vec_gt[i]，形状 (B,)
        sim_pos = (vec_gen * vec_gt).sum(dim=1)

        # 负样本相似度矩阵：锚点 i 和 batch 里每一张有雨图配对
        # sim_neg[i, j] = vec_gen[i] . vec_input[j]，形状 (B, B)
        sim_neg = vec_gen @ vec_input.t()

        # 除以温度，得到 InfoNCE 的 logits
        logits_pos = sim_pos / self.temperature                    # (B,)
        logits_neg = sim_neg / self.temperature                    # (B, B)

        # InfoNCE：-log[ exp(logits_pos) / (exp(logits_pos) + sum_j exp(logits_neg_j)) ]
        # 用 logsumexp 实现，先把正样本 logits 和全部负样本 logits 拼成一行，
        # 再算 logsumexp，避免 exp 直接溢出，数值上更稳定。
        all_logits = torch.cat([logits_pos.unsqueeze(1), logits_neg], dim=1)  # (B, 1 + B)
        loss = -logits_pos + torch.logsumexp(all_logits, dim=1)               # (B,)
        return loss.mean()
