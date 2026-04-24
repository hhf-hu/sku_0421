import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32)


def compute_phase_diff(img1, img2):
    # FFT
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)

    # 相位谱
    phase1 = np.angle(f1)
    phase2 = np.angle(f2)

    # 处理相位环绕的差值
    phase_diff = np.angle(np.exp(1j * (phase1 - phase2)))

    # 平均绝对相位差
    mean_abs_diff = np.mean(np.abs(phase_diff))

    # 相位相关峰值（作为相似性度量，峰值越高越相似）
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1) * np.abs(f2) + 1e-10)
    phase_correlation = np.abs(np.fft.ifft2(cross_power))
    peak_value = np.max(phase_correlation)

    return mean_abs_diff, peak_value


# 示例
img1 = load_image('/Users/vincent/workspace/sku/Unit0001/IMG_0937.JPG')
img2 = load_image('/Users/vincent/workspace/sku/Unit0002/IMG_0009.JPG')
mean_abs_diff, peak = compute_phase_diff(img1, img2)
print(f"平均绝对相位差: {mean_abs_diff}")
print(f"相位相关峰值: {peak}")