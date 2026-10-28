import torch
import torch.nn as nn


class DerainNet(nn.Module):
    def __init__(self, channels=64):
        super(DerainNet, self).__init__()
        C = channels

        # 输入卷积：将有雨图像从 RGB 3 通道提升到 C 通道
        self.conv_in = nn.Conv2d(3, C, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 下采样：分支 2 用一次得到 1/2 尺寸，分支 3 连用两次得到 1/4 尺寸
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 三个分支各自的 8 层残差块（单块结构与 BaselineNet 的 res_blocks 相同）
        self.branch1_blocks = self._make_res_blocks(C)
        self.branch2_blocks = self._make_res_blocks(C)
        self.branch3_blocks = self._make_res_blocks(C)

        # 上采样：分支 2 放大 2 倍，分支 3 放大 4 倍，通道数保持 C
        self.up2 = nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(C, C, kernel_size=2, stride=2),
            nn.ConvTranspose2d(C, C, kernel_size=2, stride=2),
        )

        # 融合层：拼接后的 3C 通道先压回 C 通道，再输出 3 通道的雨层
        self.conv_fuse = nn.Conv2d(3 * C, C, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(C, 3, kernel_size=3, padding=1)

    @staticmethod
    def _make_res_blocks(C):
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            for _ in range(8)
        ])

    def _run_branch(self, blocks, x):
        # 逐块处理，每块都按 BaselineNet 的方式加回本块的输入（块内残差）
        features = x
        for block in blocks:
            residual = features
            features = block(features)
            features = features + residual
        # 8 层处理完，再与整个分支处理前的特征相加（分支级跳连）
        return features + x

    def forward(self, x):
        # 保存输入的有雨图像
        rainy_image = x

        # 输入卷积，得到 C 通道特征
        features = self.relu(self.conv_in(x))

        # 分支 1：原图大小（不下采样、不上采样）
        branch1 = self._run_branch(self.branch1_blocks, features)

        # 分支 2：1/2 大小处理后，上采样回原尺寸
        branch2 = self._run_branch(self.branch2_blocks, self.pool(features))
        branch2 = self.up2(branch2)

        # 分支 3：1/4 大小处理后，上采样回原尺寸
        branch3 = self._run_branch(self.branch3_blocks, self.pool(self.pool(features)))
        branch3 = self.up4(branch3)

        # 三个分支沿通道维拼接，卷积融合后输出雨层
        fused = torch.cat([branch1, branch2, branch3], dim=1)
        fused = self.relu(self.conv_fuse(fused))
        rain_layer = self.conv_out(fused)

        # 原图减去预测雨层，得到去雨结果
        clean_image = rainy_image - rain_layer

        return clean_image
