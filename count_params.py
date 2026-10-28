# 算自建模型参数量的简单代码
# 用法（在 E:\LR6 根目录、激活 venv 后）：
#     python count_params.py
# 要求：本文件放在 E:\LR6 根目录，和 main.py 同一级，
#       这样 from models.my_model import Mymodel 才能找到模型。

import torch
from models.my_model import Mymodel


def main():
    model = Mymodel()

    # 全部参数：每个参数张量的元素个数加起来
    total = sum(p.numel() for p in model.parameters())
    # 可训练参数：requires_grad 为 True 的那些
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total} ({total / 1e6:.4f}M)")
    print(f"Trainable parameters: {trainable} ({trainable / 1e6:.4f}M)")
    # float32 一个参数占 4 字节，估算权重文件大概多大（不含优化器状态）
    print(f"Approx weight size (fp32): {total * 4 / 1024 / 1024:.2f} MB")
    print()

    # 按顶层模块分开算，一眼看出参数都花在哪了
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n} ({n / 1e6:.4f}M)")

    print()
    limit = 4_000_000
    if total <= limit:
        print(f"PASS: {total} <= {limit}，满足课程 ≤4M 要求")
    else:
        print(f"FAIL: {total} > {limit}，超出课程 ≤4M 要求，需要减通道或减块数")


if __name__ == "__main__":
    main()
