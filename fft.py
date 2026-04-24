import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess_images(img1_path, img2_path, target_size=(512, 512)):
    """加载并预处理图像"""
    print(f"加载图片1: {img1_path}")
    print(f"加载图片2: {img2_path}")

    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        raise FileNotFoundError(f"无法读取图片1: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"无法读取图片2: {img2_path}")

    print(f"图片1尺寸: {img1.shape}")
    print(f"图片2尺寸: {img2.shape}")

    # 统一尺寸
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)

    # 转换为灰度图（用于傅里叶分析）
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return img1, img2, gray1, gray2


def fft_analysis(img1, img2):
    """执行傅里叶变换分析"""
    # 转换为float32
    img1_float = img1.astype(np.float32) / 255.0
    img2_float = img2.astype(np.float32) / 255.0

    # 傅里叶变换
    fft1 = np.fft.fft2(img1_float)
    fft2 = np.fft.fft2(img2_float)

    # 移到中心
    fft1_shift = np.fft.fftshift(fft1)
    fft2_shift = np.fft.fftshift(fft2)

    # 幅度谱
    magnitude1 = np.abs(fft1_shift)
    magnitude2 = np.abs(fft2_shift)

    # 相位谱
    phase1 = np.angle(fft1_shift)
    phase2 = np.angle(fft2_shift)

    # 对数幅度谱（便于可视化）
    log_magnitude1 = np.log(magnitude1 + 1)
    log_magnitude2 = np.log(magnitude2 + 1)

    return {
        'magnitude1': magnitude1,
        'magnitude2': magnitude2,
        'log_magnitude1': log_magnitude1,
        'log_magnitude2': log_magnitude2,
        'phase1': phase1,
        'phase2': phase2,
        'fft1': fft1,
        'fft2': fft2
    }


def calculate_similarity_metrics(img1, img2, gray1, gray2, fft_results):
    """计算各种相似度指标"""
    metrics = {}

    # 1. 结构相似性指数 (SSIM)
    metrics['ssim'] = ssim(gray1, gray2)

    # 2. 均方误差 (MSE)
    metrics['mse'] = np.mean((gray1 - gray2) ** 2)

    # 3. 峰值信噪比 (PSNR)
    if metrics['mse'] == 0:
        metrics['psnr'] = float('inf')
    else:
        metrics['psnr'] = 20 * np.log10(255.0 / np.sqrt(metrics['mse']))

    # 4. 傅里叶幅度谱相关性
    mag_corr = np.corrcoef(
        fft_results['magnitude1'].flatten(),
        fft_results['magnitude2'].flatten()
    )[0, 1]
    metrics['magnitude_correlation'] = mag_corr

    # 5. 幅度谱MSE
    metrics['magnitude_mse'] = np.mean(
        (fft_results['magnitude1'] - fft_results['magnitude2']) ** 2
    )

    # 6. 相位差（标准化）
    phase_diff = np.angle(np.exp(1j * (fft_results['phase1'] - fft_results['phase2'])))
    metrics['phase_mse'] = np.mean(phase_diff ** 2)

    # 7. 频带能量分析
    def calculate_band_energy(magnitude_spectrum, frequency_band):
        """计算特定频带的能量"""
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2

        # 创建距离矩阵
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

        # 归一化距离
        normalized_dist = dist_from_center / max_dist

        # 计算频带能量
        mask = (normalized_dist >= frequency_band[0]) & (normalized_dist <= frequency_band[1])
        return np.sum(magnitude_spectrum[mask])

    # 低频 (0-0.3)
    low_freq_energy1 = calculate_band_energy(fft_results['magnitude1'], (0, 0.3))
    low_freq_energy2 = calculate_band_energy(fft_results['magnitude2'], (0, 0.3))
    metrics['low_freq_similarity'] = min(low_freq_energy1, low_freq_energy2) / max(low_freq_energy1, low_freq_energy2)

    # 中频 (0.3-0.7)
    mid_freq_energy1 = calculate_band_energy(fft_results['magnitude1'], (0.3, 0.7))
    mid_freq_energy2 = calculate_band_energy(fft_results['magnitude2'], (0.3, 0.7))
    metrics['mid_freq_similarity'] = min(mid_freq_energy1, mid_freq_energy2) / max(mid_freq_energy1, mid_freq_energy2)

    # 高频 (0.7-1.0)
    high_freq_energy1 = calculate_band_energy(fft_results['magnitude1'], (0.7, 1.0))
    high_freq_energy2 = calculate_band_energy(fft_results['magnitude2'], (0.7, 1.0))
    metrics['high_freq_similarity'] = min(high_freq_energy1, high_freq_energy2) / max(high_freq_energy1,
                                                                                      high_freq_energy2)

    # 8. 综合相似度评分（加权）
    metrics['overall_similarity'] = (
            0.2 * metrics['ssim'] +
            0.3 * (1 - np.clip(metrics['mse'] / 10000, 0, 1)) +
            0.25 * (metrics['magnitude_correlation'] + 1) / 2 +
            0.15 * metrics['low_freq_similarity'] +
            0.1 * metrics['high_freq_similarity']
    )

    return metrics


def visualize_comparison(img1, img2, gray1, gray2, fft_results, metrics):
    """可视化比较结果"""
    plt.figure(figsize=(18, 12))

    # 1. 原始彩色图像
    plt.subplot(3, 5, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1 (Color)')
    plt.axis('off')

    plt.subplot(3, 5, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2 (Color)')
    plt.axis('off')

    # 2. 灰度图像
    plt.subplot(3, 5, 3)
    plt.imshow(gray1, cmap='gray')
    plt.title('Image 1 (Gray)')
    plt.axis('off')

    plt.subplot(3, 5, 4)
    plt.imshow(gray2, cmap='gray')
    plt.title('Image 2 (Gray)')
    plt.axis('off')

    # 3. 像素差异图
    plt.subplot(3, 5, 5)
    diff = cv2.absdiff(gray1, gray2)
    plt.imshow(diff, cmap='hot')
    plt.title(f'Pixel Difference\nMax Diff: {diff.max()}')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # 4. 傅里叶幅度谱
    plt.subplot(3, 5, 6)
    plt.imshow(fft_results['log_magnitude1'], cmap='gray')
    plt.title('FFT Magnitude 1 (log)')
    plt.axis('off')

    plt.subplot(3, 5, 7)
    plt.imshow(fft_results['log_magnitude2'], cmap='gray')
    plt.title('FFT Magnitude 2 (log)')
    plt.axis('off')

    # 5. 幅度谱差异
    plt.subplot(3, 5, 8)
    mag_diff = np.abs(fft_results['magnitude1'] - fft_results['magnitude2'])
    plt.imshow(np.log(mag_diff + 1), cmap='hot')
    plt.title('Magnitude Difference')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # 6. 相位谱
    plt.subplot(3, 5, 9)
    plt.imshow(fft_results['phase1'], cmap='hsv')
    plt.title('Phase Spectrum 1')
    plt.axis('off')

    plt.subplot(3, 5, 10)
    plt.imshow(fft_results['phase2'], cmap='hsv')
    plt.title('Phase Spectrum 2')
    plt.axis('off')

    # 7. 频带能量分布
    plt.subplot(3, 5, 11)
    frequency_bands = ['Low (0-0.3)', 'Mid (0.3-0.7)', 'High (0.7-1.0)']
    similarities = [
        metrics['low_freq_similarity'],
        metrics['mid_freq_similarity'],
        metrics['high_freq_similarity']
    ]

    bars = plt.bar(frequency_bands, similarities, color=['blue', 'green', 'red'])
    plt.ylim(0, 1.1)
    plt.ylabel('Similarity Ratio')
    plt.title('Frequency Band Similarity')

    # 在柱状图上添加数值
    for bar, value in zip(bars, similarities):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    # 8. 重要指标显示
    plt.subplot(3, 5, 12)
    plt.axis('off')

    metrics_text = [
        f"=== SPATIAL DOMAIN ===",
        f"SSIM: {metrics['ssim']:.4f}",
        f"MSE: {metrics['mse']:.2f}",
        f"PSNR: {metrics['psnr']:.2f} dB" if metrics['psnr'] != float('inf') else "PSNR: ∞",
        "",
        f"=== FREQUENCY DOMAIN ===",
        f"Magnitude Correlation: {metrics['magnitude_correlation']:.4f}",
        f"Magnitude MSE: {metrics['magnitude_mse']:.2e}",
        f"Phase MSE: {metrics['phase_mse']:.2e}",
        "",
        f"=== OVERALL ===",
        f"Overall Similarity: {metrics['overall_similarity']:.4f}"
    ]

    plt.text(0.1, 0.9, '\n'.join(metrics_text), fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    # 9. 相似度评分可视化
    plt.subplot(3, 5, 13)
    metric_names = ['SSIM', 'Mag Corr', 'Low Freq', 'High Freq']
    metric_values = [
        metrics['ssim'],
        (metrics['magnitude_correlation'] + 1) / 2,
        metrics['low_freq_similarity'],
        metrics['high_freq_similarity']
    ]

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    metric_values += metric_values[:1]  # 闭合雷达图
    angles += angles[:1]

    ax = plt.subplot(3, 5, 13, projection='polar')
    ax.plot(angles, metric_values, 'o-', linewidth=2)
    ax.fill(angles, metric_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_title('Similarity Radar Chart')

    # 10. 决策判断
    plt.subplot(3, 5, 14)
    plt.axis('off')

    # 判断逻辑
    if metrics['overall_similarity'] > 0.85:
        decision = "HIGHLY SIMILAR"
        color = "green"
        details = [
            "✓ Same object/structure",
            "✓ Similar composition",
            "✓ Very likely same scene"
        ]
    elif metrics['overall_similarity'] > 0.7:
        decision = "SIMILAR"
        color = "blue"
        details = [
            "✓ Related objects",
            "✓ Similar style",
            "○ Possibly same category"
        ]
    elif metrics['overall_similarity'] > 0.5:
        decision = "MODERATELY SIMILAR"
        color = "orange"
        details = [
            "○ Some similarities",
            "○ Different but related",
            "✗ Not the same object"
        ]
    else:
        decision = "DIFFERENT"
        color = "red"
        details = [
            "✗ Different objects",
            "✗ Different composition",
            "✗ Unrelated images"
        ]

    plt.text(0.5, 0.7, decision, fontsize=16, fontweight='bold',
             color=color, ha='center', va='center')
    plt.text(0.1, 0.5, '\n'.join(details), fontsize=10,
             verticalalignment='top')

    # 11. 关键差异总结
    plt.subplot(3, 5, 15)
    plt.axis('off')

    # 分析主要差异来源
    differences = []

    if metrics['mse'] > 1000:
        differences.append("High pixel-level differences")

    if metrics['magnitude_correlation'] < 0.5:
        differences.append("Different frequency distributions")

    if metrics['high_freq_similarity'] < 0.6:
        differences.append("Different details/edges")

    if metrics['low_freq_similarity'] < 0.8:
        differences.append("Different overall structure")

    if not differences:
        differences.append("Images are very similar")

    plt.text(0.1, 0.9, "Key Differences:", fontsize=11, fontweight='bold')
    for i, diff in enumerate(differences):
        plt.text(0.1, 0.8 - i * 0.1, f"• {diff}", fontsize=10)

    plt.suptitle('Image Similarity Analysis using Fourier Transform', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def main():
    # 你的图片路径
    img1_path = "/Users/vincent/workspace/sku/Unit0001/IMG_0938.JPG"
    img2_path = "/Users/vincent/workspace/sku/Unit0002/IMG_0009.JPG"

    try:
        # 1. 加载并预处理图像
        print("正在加载和处理图像...")
        img1, img2, gray1, gray2 = load_and_preprocess_images(img1_path, img2_path)

        # 2. 傅里叶变换分析
        print("正在进行傅里叶变换分析...")
        fft_results = fft_analysis(gray1, gray2)

        # 3. 计算相似度指标
        print("正在计算相似度指标...")
        metrics = calculate_similarity_metrics(img1, img2, gray1, gray2, fft_results)

        # 4. 打印详细结果
        print("\n" + "=" * 60)
        print("图像相似度分析结果")
        print("=" * 60)

        print(f"\n空间域分析:")
        print(f"  SSIM (结构相似性): {metrics['ssim']:.4f} (1.0=完全相同)")
        print(f"  MSE (均方误差): {metrics['mse']:.2f} (0=完全相同)")
        print(f"  PSNR (峰值信噪比): {metrics['psnr']:.2f} dB (越高越好)")

        print(f"\n频率域分析:")
        print(f"  幅度谱相关性: {metrics['magnitude_correlation']:.4f} (1.0=完全相同)")
        print(f"  幅度谱MSE: {metrics['magnitude_mse']:.2e}")
        print(f"  相位谱MSE: {metrics['phase_mse']:.2e}")

        print(f"\n频带相似度:")
        print(f"  低频 (0-30%): {metrics['low_freq_similarity']:.4f}")
        print(f"  中频 (30-70%): {metrics['mid_freq_similarity']:.4f}")
        print(f"  高频 (70-100%): {metrics['high_freq_similarity']:.4f}")

        print(f"\n总体相似度评分: {metrics['overall_similarity']:.4f}")
        print("=" * 60)

        # 5. 可视化
        print("\n生成可视化报告...")
        visualize_comparison(img1, img2, gray1, gray2, fft_results, metrics)

        # 6. 给出结论
        print("\n结论:")
        if metrics['overall_similarity'] > 0.85:
            print("这两张图片非常相似，很可能是同一物体/场景的不同角度或时间拍摄")
        elif metrics['overall_similarity'] > 0.7:
            print("这两张图片较为相似，可能是同类物体或具有相似特征")
        elif metrics['overall_similarity'] > 0.5:
            print("这两张图片有部分相似之处，但不是同一物体")
        else:
            print("这两张图片差异较大，是不同物体/场景")

    except Exception as e:
        print(f"错误: {e}")
        print("请检查图片路径是否正确，或安装必要的库:")
        print("pip install opencv-python numpy matplotlib scikit-image")


if __name__ == "__main__":
    main()