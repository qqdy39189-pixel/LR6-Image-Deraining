import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        # 加载在 ImageNet 上预训练的 VGG19
        weights = VGG19_Weights.DEFAULT
        vgg = vgg19(weights=weights)

        # 只取特征提取部分的前 35 层（索引 0~34），正好到最后一个卷积层为止
        self.features = vgg.features[:35]

        # 冻结 VGG19 参数，不参与训练
        for param in self.features.parameters():
            param.requires_grad = False

        # 固定为评估模式
        self.features.eval()

        # 使用 ImageNet 的均值和标准差进行归一化
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # 按本节要求：使用 L1 计算特征差异
        self.criterion = nn.L1Loss()

    def forward(self, generated, ground_truth):
        # 将图像归一化到 VGG19 预训练模型适用的输入分布
        generated = (generated - self.mean) / self.std
        ground_truth = (ground_truth - self.mean) / self.std

        # 两张图走同一条（截断后的）特征提取流水线，各得到一份深层特征
        generated_features = self.features(generated)
        ground_truth_features = self.features(ground_truth)

        # 在特征空间里量两张图的差距
        loss = self.criterion(generated_features, ground_truth_features)

        return loss
