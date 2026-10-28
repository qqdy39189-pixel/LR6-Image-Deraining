import torch
import torch.nn as nn


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        # 三层卷积串联：RGB 3 通道 -> 16 通道 -> 16 通道 -> RGB 3 通道
        # 每层 padding=1，特征图长宽始终不变
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # 直接输出预测的去雨图像
        return self.features(x)
