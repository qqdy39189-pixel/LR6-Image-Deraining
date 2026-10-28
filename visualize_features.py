import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 第 1 步：选一张测试图

img = Image.open(r"E:\LR6\data\test\input\1.png").convert("RGB")
img.save("sample_input.png")

# 第 2 步：按预训练模型的标准喂法预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
x = preprocess(img).unsqueeze(0)   # (3,224,224) -> (1,3,224,224)

# 第 3 步：加载预训练 AlexNet，只要特征提取部分
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
features = model.features
features.eval()

# 第 4 步：逐层往前传，截获 5 个卷积层的输出
capture_at = {0: "Conv1", 3: "Conv2", 6: "Conv3", 8: "Conv4", 10: "Conv5"}
captured = []
with torch.no_grad():
    out = x
    for idx, layer in enumerate(features):
        out = layer(out)
        if idx in capture_at:
            captured.append((capture_at[idx], out[0].clone()))

# 第 5 步：把特征图画成网格
fig, axes = plt.subplots(len(captured), 8, figsize=(16, 2.2 * len(captured)))
for row, (name, fmap) in enumerate(captured):
    for col in range(8):
        ax = axes[row][col]
        channel = fmap[col]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        ax.imshow(channel.numpy(), cmap="gray")
        ax.axis("off")
        if col == 0:
            ax.set_title(f"{name}\n{tuple(fmap.shape)}", fontsize=9)
    print(name, "feature shape:", tuple(fmap.shape))
plt.tight_layout()
plt.savefig("feature_viz.png", dpi=150)
print("Saved: feature_viz.png")
