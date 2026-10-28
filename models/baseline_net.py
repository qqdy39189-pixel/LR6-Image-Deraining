import torch
import torch.nn as nn


class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()

        # 第 1 层：输入有雨图像，RGB 3 通道 -> 64 通道
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 第 2～7 层：6 个 64 通道卷积层
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            for _ in range(6)
        ])

        # 第 8 层：预测雨层，64 通道 -> RGB 3 通道
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 保存输入的有雨图像
        rainy_image = x

        # 提取初始特征
        features = self.relu(self.conv1(x))

        # 使用残差连接提取深层特征
        for block in self.res_blocks:
            residual = features
            features = block(features)
            features = features + residual

        # 预测雨层
        rain_layer = self.conv_out(features)

        # 原图减去预测雨层，得到去雨结果
        clean_image = rainy_image - rain_layer

        return clean_image
