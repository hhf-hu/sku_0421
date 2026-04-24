#!/usr/bin/env python3
# convert_all_grayscale.py

import os
import sys
import glob
from PIL import Image
from pathlib import Path
import argparse
import concurrent.futures
from tqdm import tqdm  # 进度条，可选安装


def get_all_image_files(root_dir, extensions=None):
    """递归获取所有图片文件"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif',
                      '.JPG', '.JPEG', '.PNG', '.BMP', '.GIF', '.TIFF', '.TIF']

    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                image_files.append(os.path.join(root, file))

    return image_files


def convert_to_grayscale(input_path, output_path=None, overwrite=False):
    """将单张图片转为灰度图"""
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 转为灰度图
            gray_img = img.convert('L')

            # 转为RGB模式（三通道灰度图）
            gray_rgb = gray_img.convert('RGB')

            # 确定输出路径
            if output_path:
                final_output = output_path
            elif overwrite:
                final_output = input_path
            else:
                # 在原目录添加_gray后缀
                base, ext = os.path.splitext(input_path)
                final_output = f"{base}_gray{ext}"

            # 保存图片
            gray_rgb.save(final_output)
            return True, final_output

    except Exception as e:
        return False, f"处理 {input_path} 时出错: {str(e)}"


def process_batch(input_dir, output_dir=None, overwrite=False, workers=4):
    """批量处理图片"""
    print("🔍 正在扫描目录结构...")

    # 获取所有图片文件
    image_files = get_all_image_files(input_dir)

    if not image_files:
        print("❌ 没有找到图片文件")
        return 0, 0

    print(f"📁 共找到 {len(image_files)} 张图片")
    print("🔄 开始转换为灰度图...")

    success_count = 0
    failed_files = []

    # 创建输出目录结构
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 使用进度条
    with tqdm(total=len(image_files), desc="转换进度", unit="张") as pbar:
        # 并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for img_path in image_files:
                # 计算输出路径
                if output_dir:
                    rel_path = os.path.relpath(img_path, input_dir)
                    out_path = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                else:
                    out_path = None

                # 提交任务
                future = executor.submit(convert_to_grayscale, img_path, out_path, overwrite)
                futures.append(future)

            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                success, result = future.result()
                if success:
                    success_count += 1
                else:
                    failed_files.append(result)
                pbar.update(1)
                pbar.set_postfix({"成功": success_count, "失败": len(failed_files)})

    return success_count, failed_files


def generate_report(input_dir, success_count, failed_files, output_dir=None):
    """生成转换报告"""
    print("\n" + "=" * 60)
    print("📊 转换完成！统计报告")
    print("=" * 60)

    total_files = success_count + len(failed_files)

    print(f"📁 输入目录: {input_dir}")
    if output_dir:
        print(f"📂 输出目录: {output_dir}")
    else:
        print("📝 模式: 原地转换" if overwrite else "📝 模式: 在原目录创建灰度副本")

    print(f"📈 总图片数: {total_files}")
    print(f"✅ 成功转换: {success_count}")
    print(f"❌ 转换失败: {len(failed_files)}")

    if total_files > 0:
        print(f"📊 成功率: {success_count / total_files * 100:.1f}%")

    if failed_files:
        print(f"\n📝 失败文件列表:")
        for i, error in enumerate(failed_files[:10], 1):  # 只显示前10个
            print(f"  {i}. {error}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个失败文件")

        # 保存失败日志
        log_file = "conversion_errors.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            for error in failed_files:
                f.write(error + '\n')
        print(f"\n📄 详细错误日志已保存到: {log_file}")


def main():
    parser = argparse.ArgumentParser(description='递归遍历所有子文件夹并转换图片为灰度图')
    parser.add_argument('input', help='输入目录路径')
    parser.add_argument('-o', '--output', help='输出目录路径（可选，不指定则使用原地模式）')
    parser.add_argument('-f', '--force', action='store_true',
                        help='覆盖原文件（仅当不使用-o参数时有效）')
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='并行工作线程数（默认: 4）')
    parser.add_argument('--ext', default='jpg,jpeg,png,bmp,gif,tiff,tif',
                        help='支持的图片扩展名，逗号分隔（默认: jpg,jpeg,png,bmp,gif,tiff,tif）')

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.isdir(args.input):
        print(f"❌ 错误: 目录不存在: {args.input}")
        return

    # 检查是否覆盖原文件
    overwrite = args.force
    if overwrite and args.output:
        print("⚠️  警告：同时使用 -o 和 -f 参数，将忽略 -f 参数")
        overwrite = False

    if overwrite and not args.output:
        confirm = input("⚠️  警告：这将覆盖所有原始图片！确认继续？(y/n): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return

    print("=" * 60)
    print("🖼️  批量图片灰度转换工具")
    print("=" * 60)

    # 执行转换
    success, failed = process_batch(
        args.input,
        args.output,
        overwrite,
        args.workers
    )

    # 生成报告
    generate_report(args.input, success, failed, args.output)


if __name__ == "__main__":
    main()