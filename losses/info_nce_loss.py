import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        self.temperature = temperature

        # 与感知损失、Easy 对比损失相同的预训练模型、相同的中间层：
        weights = VGG19_Weights.DEFAULT
        vgg = vgg19(weights=weights)
        self.features = vgg.features[:35]

        # 冻结 VGG19 参数，只当特征提取器，不参与训练
        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()

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
        # 归一化后，向量点积就等于余弦相似度。
        image = (image - self.mean) / self.std
        feat_map = self.features(image)     
        pooled = self.pool(feat_map)       
        vec = pooled.flatten(1)            
        return F.normalize(vec, p=2, dim=1)  

    def forward(self, input_image, generated, ground_truth):
        # 三张图走同一条流水线，各得到一条单位向量
        vec_input = self._to_vector(input_image)   
        vec_gen = self._to_vector(generated)       
        vec_gt = self._to_vector(ground_truth)     

        sim_pos = (vec_gen * vec_gt).sum(dim=1)

        sim_neg = vec_gen @ vec_input.t()

        # 除以温度，得到 InfoNCE 的 logits
        logits_pos = sim_pos / self.temperature
        logits_neg = sim_neg / self.temperature

        # 用 logsumexp 实现，先把正样本 logits 和全部负样本 logits 拼成一行，
        # 再算 logsumexp，避免 exp 直接溢出，数值上更稳定。
        all_logits = torch.cat([logits_pos.unsqueeze(1), logits_neg], dim=1)
        loss = -logits_pos + torch.logsumexp(all_logits, dim=1)
        return loss.mean()
