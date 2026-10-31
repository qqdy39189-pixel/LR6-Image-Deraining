import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class EasyContrastiveLoss(nn.Module):
    def __init__(self):
        super(EasyContrastiveLoss, self).__init__()

        # 与感知损失相同的预训练模型、相同的中间层：
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
        image = (image - self.mean) / self.std
        feat_map = self.features(image) 
        pooled = self.pool(feat_map) 
        return pooled.flatten(1)  

    def forward(self, input_image, generated, ground_truth):
        # 三张图走同一条流水线，各得到一条特征向量
        vec_input = self._to_vector(input_image)  
        vec_gen = self._to_vector(generated)     
        vec_gt = self._to_vector(ground_truth)   

        # 余弦相似度：只比方向，取值在 -1 到 1 之间
        sim_pos = F.cosine_similarity(vec_gen, vec_gt, dim=1)     # 预测 vs 目标，想让它大
        sim_neg = F.cosine_similarity(vec_gen, vec_input, dim=1)  # 预测 vs 输入，想让它小

        # 最小化这个损失 <=> 拉近预测与目标、推远预测与有雨输入
        loss = (1.0 - sim_pos) + sim_neg
        return loss.mean()
