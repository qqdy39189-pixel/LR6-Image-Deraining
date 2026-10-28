import torch
import torch.nn as nn


class DerainNet(nn.Module):
    def __init__(self):
        super(DerainNet, self).__init__()

        # 第 1 层：输入有雨图像，RGB 3 通道 -> 32 通道
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 第 2 层：32 通道卷积，继续提取特征
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # 第 3 层：预测雨层，32 通道 -> RGB 3 通道
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 保存输入的有雨图像
        rainy_image = x

        # 提取特征
        features = self.relu(self.conv1(x))
        features = self.relu(self.conv2(features))

        # 预测雨层
        rain_layer = self.conv_out(features)

        # 原图减去预测雨层，得到去雨结果
        clean_image = rainy_image - rain_layer

        return clean_image
