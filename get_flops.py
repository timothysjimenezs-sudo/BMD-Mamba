import torch
import os
from thop import profile
from models import build_model

# 伪装参数类，完美避开 argparse 冲突
class DummyArgs:
    def __init__(self):
        self.Norm_Type = 'GN'
        self.dataset_mode = 'crack'
        self.device = 'cuda'  # 强制使用 GPU
        self.BCELoss_ratio = 0.83
        self.DiceLoss_ratio = 0.17

def main():
    print("⏳ 正在构建模型 (GPU 模式)...")
    args = DummyArgs()
    
    # 构建模型并放入显卡
    model, _ = build_model(args)
    model.cuda()
    model.eval()

    print("⏳ 正在计算 Params 和 FLOPs (标准输入 512x512)...")
    # 构造一张 512x512 的伪造图片放入显卡
    dummy_input = torch.randn(1, 3, 512, 512).cuda()

    # 使用 thop 计算 MACs (乘加操作数) 和 Params
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)

    print("⏳ 正在计算真实的 Model Size...")
    # 将模型权重物理保存到硬盘，测完大小后再删掉，这是最准的 Size 测法
    temp_weight_path = 'temp_model_size_test.pth'
    torch.save(model.state_dict(), temp_weight_path)
    size_mb = os.path.getsize(temp_weight_path) / (1024 * 1024)
    os.remove(temp_weight_path) # 测完清理垃圾

    print("\n" + "="*45)
    print("✨ 计算完成！请将以下确切数据填入 Table 3：")
    # 学术界通常将 FLOPs 视为 MACs 的 2 倍
    print(f"Params ↓      : {params / 1e6:.2f} M")
    print(f"FLOPs  ↓      : {macs * 2 / 1e9:.2f} G")
    print(f"Model Size ↓  : {size_mb:.0f} MB")
    print("="*45)

if __name__ == '__main__':
    main()