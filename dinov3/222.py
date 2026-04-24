import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path

# 定义文件路径
file_path = "/Users/vincent/workspace/sku/matrix/dinov3-vith16plus-pretrain-lvd1689m_similarity_jojo_data-1231-6_results.pt"
# 如果你有预测结果文件，也可以一起加载
predictions_file = "predictions_dinov3-vith16plus-pretrain-lvd1689m_similarity_jojo_data-1231-4.csv"

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"文件不存在: {file_path}")
    exit(1)

print(f"正在加载文件: {file_path}")

try:
    # 加载文件
    results = torch.load(file_path, map_location='cpu', weights_only=False)

    print("\n" + "=" * 50)
    print("文件加载成功！")
    print("=" * 50)

    # 1. 显示所有键
    print("\n1. 文件包含的键:")
    for key in results.keys():
        if isinstance(results[key], (list, np.ndarray)):
            print(f"  - {key}: {type(results[key]).__name__} [长度: {len(results[key])}]")
        elif isinstance(results[key], dict):
            print(f"  - {key}: dict [大小: {len(results[key])}]")
        else:
            print(f"  - {key}: {type(results[key]).__name__}")

    # 2. 显示标签映射
    if 'label_mapping' in results:
        print("\n2. 标签映射:")
        for label, idx in results['label_mapping'].items():
            print(f"  '{label}' -> {idx}")

    # 3. 尝试加载预测结果文件
    predictions_data = None
    if os.path.exists(predictions_file):
        print(f"\n3. 找到预测文件: {predictions_file}")
        predictions_df = pd.read_csv(predictions_file)
        print(f"   预测文件包含 {len(predictions_df)} 行")
        print(f"   列名: {list(predictions_df.columns)}")
        predictions_data = predictions_df

    # 4. 创建详细结果CSV
    print("\n4. 创建详细结果CSV...")

    if 'image_paths' in results and 'labels' in results:
        data = []

        for i in range(len(results['image_paths'])):
            img_path = results['image_paths'][i]
            true_label_id = results['labels'][i]

            # 获取真实标签名称
            true_label_name = None
            if 'label_mapping' in results:
                for name, idx in results['label_mapping'].items():
                    if idx == true_label_id:
                        true_label_name = name
                        break

            # 初始化预测相关字段
            predicted_label_id = None
            predicted_label_name = None
            similarity = None
            is_correct = "未知"

            # 如果存在预测文件，尝试匹配预测结果
            if predictions_data is not None:
                # 尝试通过文件名匹配
                filename = os.path.basename(img_path)

                # 查找匹配的行（可能有多种匹配方式）
                matches = predictions_data[predictions_data['file_name'] == filename]
                if len(matches) == 0:
                    # 尝试通过完整路径匹配
                    matches = predictions_data[predictions_data['image_path'] == img_path]

                if len(matches) > 0:
                    # 使用第一个匹配结果
                    match = matches.iloc[0]
                    predicted_label_id = match.get('predicted_label_id')
                    if predicted_label_id is None:
                        # 如果没有predicted_label_id，尝试从predicted_label推断
                        pred_label = match.get('predicted_label')
                        if pred_label in results['label_mapping']:
                            predicted_label_id = results['label_mapping'][pred_label]

                    similarity = match.get('similarity')

                    # 如果有预测标签ID，获取标签名称
                    if predicted_label_id is not None and 'label_mapping' in results:
                        for name, idx in results['label_mapping'].items():
                            if idx == predicted_label_id:
                                predicted_label_name = name
                                break

                    # 判断是否预测正确
                    if predicted_label_id is not None:
                        is_correct = "是" if true_label_id == predicted_label_id else "否"

            data.append({
                '序号': i + 1,
                '文件名': os.path.basename(img_path),
                '完整路径': img_path,
                '真实标签ID': true_label_id,
                '真实标签名称': true_label_name,
                '预测标签ID': predicted_label_id,
                '预测标签名称': predicted_label_name,
                '相似度': similarity,
                '是否预测正确': is_correct
            })

        # 创建DataFrame
        df = pd.DataFrame(data)

        # 统计预测正确率（如果有预测数据）
        if predictions_data is not None:
            correct_count = df[df['是否预测正确'] == '是'].shape[0]
            total_with_pred = df[df['是否预测正确'] != '未知'].shape[0]
            if total_with_pred > 0:
                accuracy = correct_count / total_with_pred
                print(f"   预测正确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
                print(f"   正确数量: {correct_count}/{total_with_pred}")

        # 保存CSV
        csv_path = file_path.replace('.pt', '_detailed_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   详细结果已保存到: {csv_path}")
        print(f"   总行数: {len(df)}")

        # 显示前几行
        print("\n5. 前5行结果:")
        print(df.head().to_string())

    else:
        print("错误: 文件缺少必要的键（image_paths 或 labels）")

except Exception as e:
    print(f"\n加载失败: {e}")
    print("\n尝试其他加载方式...")

    try:
        # 尝试使用 add_safe_globals
        import numpy._core.multiarray

        torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
        results = torch.load(file_path, map_location='cpu', weights_only=True)
        print("使用 add_safe_globals 加载成功！")
    except Exception as e2:
        print(f"再次加载失败: {e2}")