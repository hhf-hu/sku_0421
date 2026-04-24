import os
import torch
import shutil


# 保存检查点函数
def save_checkpoint(checkpoint, is_best_f1, is_best_loss, is_best_pr_auc, is_best_similarity,
                    checkpoint_path, best_f1_path, best_loss_path, best_pr_auc_path, best_similarity_path):
    # 保存常规检查点
    torch.save(checkpoint, checkpoint_path)

    # 保存最佳F1模型
    if is_best_f1:
        torch.save(checkpoint, best_f1_path)

    # 保存最佳损失模型
    if is_best_loss:
        torch.save(checkpoint, best_loss_path)

    # 保存最佳PR-AUC模型
    if is_best_pr_auc:
        torch.save(checkpoint, best_pr_auc_path)

    # 保存最佳相似度模型
    if is_best_similarity:
        torch.save(checkpoint, best_similarity_path)


# 加载检查点函数
def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    if os.path.isfile(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 处理状态字典的键名不匹配问题
        state_dict = checkpoint['state_dict']

        # 检查模型是否是DeepSpeed包装的
        if hasattr(model, 'module'):
            # 如果当前使用DeepSpeed但检查点没有module前缀，需要添加前缀
            if not any(key.startswith('module.') for key in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k if not k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
        else:
            # 如果当前不使用DeepSpeed包装但检查点有module前缀，需要移除前缀
            if any(key.startswith('module.') for key in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict

        # 加载状态字典
        model.load_state_dict(state_dict)

        # 加载优化器和调度器状态
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint.get('best_f1', 0.0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        best_pr_auc = checkpoint.get('best_pr_auc', 0.0)

        print(
            f"从 epoch {start_epoch} 继续训练，最佳 F1: {best_f1:.4f}, 最佳损失: {best_loss:.4f}, 最佳PR-AUC: {best_pr_auc:.4f}")
        return start_epoch, best_f1, best_loss, best_pr_auc
    else:
        print(f"未找到检查点: {checkpoint_path}，从头开始训练")
        return 0, 0.0, float('inf'), 0.0