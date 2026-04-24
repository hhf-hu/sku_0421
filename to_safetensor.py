import torch
from safetensors.torch import save_file


def convert_pth_to_safetensors(pth_path, safetensors_path):
    # 加载原始模型
    checkpoint = torch.load(pth_path, map_location='cpu',weights_only=False)

    # 检查checkpoint的结构
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 处理状态字典键名（移除可能的'module.'前缀）
        if any(key.startswith('module.') for key in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict


    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print(f"模型包含 {len(state_dict)} 个参数")

    # 保存为safetensors格式
    save_file(state_dict, safetensors_path)
    print(f"转换完成！保存为: {safetensors_path}")


# 使用示例
convert_pth_to_safetensors(
    'model_best_similarity0917.pth.tar',
    'model_best_similarity0917.safetensors'
)