# 把自建模型导出为 ONNX 文件，再去 netron.app 看结构
# 用法（在 E:\LR6 根目录、激活 venv 后）：
#     python export_onnx.py
#     # 成功后会在当前目录生成 mymodel.onnx
#     # 打开 https://netron.app ，把 mymodel.onnx 拖进去即可看结构图
# 要求：本文件放在 E:\LR6 根目录，和 main.py 同一级。
# 注意：如果报错 No module named 'onnx'，先装一次：
#     pip install onnx

import torch
from models.my_model import Mymodel


def main():
    model = Mymodel()
    model.eval()  # 导出时固定为推理模式（关掉训练时的随机行为，本模型无 dropout/BN，影响不大但这是习惯）

    # 假输入：只用来“走一遍图、把结构定下来”，里面的数值不重要
    # 形状 (1, 3, 256, 256) = 1 张图、3 通道、256x256
    # 模型是全卷积的，换成 64x64 也能导出，图结构一样
    dummy_input = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model,
        dummy_input,
        "mymodel.onnx",
        input_names=["rainy"],    # 输入在 netron 里显示的名字
        output_names=["clean"],   # 输出在 netron 里显示的名字
        # batch、高、宽设为动态轴：导出后换 batch 数、换图尺寸都能用同一张图表示
        dynamic_axes={
            "rainy": {0: "batch", 2: "height", 3: "width"},
            "clean": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=11,         # 兼容性较好的版本，netron 支持
        do_constant_folding=True,  # 把常量折叠掉，图更干净
        dynamo=False,             # 用传统导出器，得到的图和常见的 ONNX 结构一致
    )
    print("Saved to mymodel.onnx")


if __name__ == "__main__":
    main()
