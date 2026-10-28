import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    残差块(ResBlock)。

    思路：
        不让网络直接学“从有雨图到干净图”的完整映射，
        而是学一个修正量，再把修正量加回输入。
        这样梯度可以直接沿直通路回传，不容易在深层消失，
        训练也更稳。
    作用：
        主干的基本重复单元，负责在 64 通道、原分辨率上
        提取雨纹和背景的空间特征，尺寸全程不变。
    参考论文：
        PReNet(Ren et al., arXiv:1901.09221)中的 ResBlock:
        Conv -> ReLU -> Conv,输出再与本块输入相加、最后过一次 ReLU。
        本实现与之结构一致，只是通道数按自建模型定为 64。
        BaselineNet / DerainNet 课程代码里的残差块也是同一思想。
    """

    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    """
    通道注意力(Channel Attention)。

    思路：
        不是所有通道都一样重要。先把每张特征图在空间上平均成
        一个数（全局平均池化），得到“每个通道整体有多强”，
        再用两层 1x1 卷积学一组逐通道权重(0~1 之间），
        重要的通道放大，不重要的通道压小。
    作用：
        回答“什么重要”(what):在通道维度上重标定特征，
        让网络把算力集中到对去雨更有用的通道上。
    参考论文：
        Mehri 等 MPRNet(WACV 2021,arXiv:2011.04566)中
        TFAM 的通道单元(Channel Unit):全局平均池化得一阶统计，
        再用并行卷积重标定通道。本实现做了简化：
        用两层 1x1 卷积 + 缩放比 4 代替原文的分半并行卷积，
        结构更简单、参数更少，思想一致。
        另可对照 RCAN 的通道注意力（仅通道、不含空间）。
    """

    def __init__(self, channels=64, reduction=4):
        super(ChannelAttention, self).__init__()
        mid = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (B, C, H, W) -> (B, C, 1, 1)，再广播乘回原特征
        weight = self.sigmoid(self.fc(self.avg_pool(x)))
        return x * weight


class SpatialAttention(nn.Module):
    """
    空间注意力(Spatial Attention)。

    思路：
        雨纹在图上的位置是不均匀的。把特征在通道维度上
        分别做平均池化和最大池化，得到两张单通道的空间图：
        平均图反映“这片区域整体什么样”，最大图反映
        “这片区域最突出的点在哪”，把两张图拼起来，
        再用一层 7x7 卷积学一张空间权重图(0~1 之间)。
    作用：
        回答“哪里重要”(where):在空间维度上重标定特征，
        让网络重点关注有雨的区域，少关注干净背景。
    参考论文：
        SPANet(Wang et al., arXiv:1904.01538)中 SAM 的思想：
        用注意力图引导残差块去雨，注意力图乘残差分支。
        本实现做了简化：不用 SPANet 的 IRNN 四方向递推
        （复现成本高、老代码依赖多），改用平均 + 最大池化
        拼描述子 + 7x7 卷积的轻量版，思想一致、实现更稳。
        另可对照 Mehri MPRNet 中 TFAM 的位置单元(Pos Unit)。
    注意：
        本模块的注意力图没有单独的监督信号（不像 SPANet
        那样用二值雨图监督第一张注意力图），权重是靠最终
        的去雨损失端到端学出来的。
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维平均、沿通道维取最大，各得 (B, 1, H, W)，拼成 (B, 2, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        weight = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        # (B, 1, H, W) 广播乘到全部通道
        return x * weight


class DualAttention(nn.Module):
    """
    双重注意力(Dual Attention,先通道、后空间)。

    思路：
        通道注意力和空间注意力是互补的：一个管“哪个通道重要”，
        一个管“哪个位置重要”。先做通道重标定，再做空间重标定，
        两步串联，比只做其中一步覆盖更全。
    作用：
        放在主干残差块之后、输出层之前，对主干特征做一次
        整体精炼，相当于 Mehri MPRNet 里特征模块(FM)中
        TFAM 的位置：残差模块之后、重建之前。
    参考论文：
        Mehri 等 MPRNet(WACV 2021)TFAM:通道单元 + 位置单元
        双路汇合、1x1 恢复通道、sigmoid 出掩码、乘特征、加直通。
        本实现是 TFAM 的简化串联版：通道、空间各做一次，
        不做双路汇合后的 1x1 融合，参数更少、流程更清晰。
    """

    def __init__(self, channels=64):
        super(DualAttention, self).__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class Mymodel(nn.Module):
    """
    自建模型 Mymodel。

    整体思路（不是照搬任何一篇的整架构）：
        1. 浅层卷积先把 3 通道雨图升到 64 通道特征；
        2. 6 个残差块在原分辨率上反复提炼特征(PReNet 式残差块);
        3. 双重注意力对特征做通道 + 空间精炼(SPANet 空间思想
           + Mehri TFAM 通道 + 空间思想的简化版);
        4. 长程跳连：把精炼后的特征与浅层特征相加，
           保证原始浅层信息不丢(Mehri LRSC 的简化版，
           这里是直接相加，不是拼接 + 1x1 融合);
        5. 输出层预测雨层，原图减雨层得到干净图
           (PReNet / DerainNet 的负残差思想：学雨层比
           直接学干净图更容易，梯度也有直通路)。

    与课程 main.py 的接口：
        无参构造 Mymodel(),forward 输入 (B, 3, H, W) 的雨图，
        输出同尺寸 (B, 3, H, W) 的去雨图,不用改 main.py、不用改损失、
        不用改存档逻辑(存档名自动为 best_mymodel.pth.tar)。

    参数量：约 0.45M,远低于课程 ≤4M 上限（见下一步
        参数量代码的实测结果）。
    训练设置：
        像素损失 MSE(L2),Adam,main.py 默认超参数，
        训练 200 轮，测试时不加任何损失开关。
    """

    def __init__(self):
        super(Mymodel, self).__init__()
        # 浅层特征提取：3 -> 64，尺寸不变
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 主干：6 个残差块串联
        self.body = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # 精炼：双重注意力
        self.attention = DualAttention(64)

        # 输出层：预测雨层，64 -> 3
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        rainy_image = x

        # 浅层特征
        shallow = self.relu(self.conv_in(x))

        # 主干提炼
        features = self.body(shallow)

        # 注意力精炼
        features = self.attention(features)

        # 长程跳连：精炼特征 + 浅层特征，保原始信息
        features = features + shallow

        # 预测雨层，原图减雨层
        rain_layer = self.conv_out(features)
        clean_image = rainy_image - rain_layer
        return clean_image
