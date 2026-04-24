import copy
import os

import matplotlib.pyplot as plt

os.chdir("../..")
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
# from src.utils.plotting import make_matching_figure
#
# from util_Files import *
# from src.loftr import LoFTR, default_cfg

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# def marchenko_pastur_pdf(x, gamma):
#     """
#     Marchenko-Pastur PDF for eigenvalues of (1/n) X^T X,
#     where X is n x p, gamma = p/n <= 1.
#     For gamma > 1, there's a point mass at zero, but we focus on gamma <= 1 here.
#     We handle gamma > 1 by using c = 1/gamma and swapping roles.
#     """
#     if gamma > 1:
#         # Use dual: non-zero eigenvalues same as for gamma' = 1/gamma < 1
#         c = 1.0 / gamma
#         lower = (1 - np.sqrt(c))**2
#         upper = (1 + np.sqrt(c))**2
#         pdf = np.sqrt((upper - x) * (x - lower)) / (2 * np.pi * c * x)
#         pdf = np.where((x >= lower) & (x <= upper), pdf, 0.0)
#         return pdf
#     else:
#         c = gamma
#         lower = (1 - np.sqrt(c))**2
#         upper = (1 + np.sqrt(c))**2
#         pdf = np.sqrt((upper - x) * (x - lower)) / (2 * np.pi * c * x)
#         pdf = np.where((x >= lower) & (x <= upper), pdf, 0.0)
#         return pdf
def compute_KL(simi_original_, mask_):
    simi_original = copy.deepcopy(simi_original_)
    mask = copy.deepcopy(mask_)

    for ii in range(mask.shape[0]):
        for jj in range(mask.shape[1]):
            if ii>=jj:
                mask[ii, jj]=3

    intersamples = (simi_original[mask == 1])
    intrasamples = (simi_original[mask == 0])
    # 1. Define common bins (important!)
    bins = np.linspace(
        min(np.min(intersamples), np.min(intrasamples)),
        max(np.max(intersamples), np.max(intrasamples)),
        61  # 60 bins → 61 edges
    )

    # 2. Compute histograms with density=True and same bins
    count_inter, _ = np.histogram(intersamples, bins=bins, density=True)
    count_intra, _ = np.histogram(intrasamples, bins=bins, density=True)

    # Avoid log(0): add small epsilon or mask zeros
    eps = 1e-10
    p = np.clip(count_inter, eps, None)
    q = np.clip(count_intra, eps, None)

    kl_pq = np.sum(p * np.log(p / q)) * np.diff(bins)[0]  # KL(P || Q)
    kl_qp = np.sum(q * np.log(q / p)) * np.diff(bins)[0]  # KL(Q || P)
    js = 0.5 * (kl_pq + kl_qp)  # Jensen-Shannon divergence

    print(f"KL: {kl_pq:.4f}, JS: {js:.4f}")

    print(rf"Number of mask: {np.sum(mask)}/{mask.shape[0]*mask.shape[1]}")

    simi_same = simi_original[mask==1]
    simi_diff = simi_original[mask==0]
    print(rf"Number of true > 0.5: {np.sum(simi_same>0.5)}/{np.sum(mask)}")
    print(rf"Number of false > 0.5: {np.sum(simi_diff > 0.5)}/{np.sum(mask==0)-np.sum(mask)}")

    thres = 0.6
    print(rf"Number of true > {thres}: {np.sum(simi_same > thres)}/{np.sum(mask)}")
    print(rf"Number of false > {thres}: {np.sum(simi_diff > thres)}/{np.sum(mask == 0) - np.sum(mask)}")



def pr_curve(simi_original_, mask_):
    simi_original = copy.deepcopy(simi_original_)
    mask = copy.deepcopy(mask_)
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc

    for ii in range(mask.shape[0]):
        for jj in range(mask.shape[1]):
            if ii>=jj:
                mask[ii, jj]=3

    pos_scores_ori = list(simi_original[mask == 1])
    neg_scores_ori = list(simi_original[mask == 0])

    neg_scores = neg_scores_ori
    pos_scores = list(np.random.choice(pos_scores_ori, size=len(neg_scores_ori), replace=True))
    # balanced_neg_indices = np.random.choice(len(neg_scores), size=len(pos_scores), replace=False)
    # neg_scores = list(np.array(neg_scores)[balanced_neg_indices])


    # 合并：正样本标签为1，负样本为0
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = pos_scores + neg_scores

    y_true_ori = [1] * len(pos_scores_ori) + [0] * len(neg_scores_ori)
    y_scores_ori = pos_scores_ori + neg_scores_ori

    # 计算 Precision-Recall 曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision_ori, recall_ori, thresholds_ori = precision_recall_curve(y_true_ori, y_scores_ori)

    # precision = precision[10:-10]
    # recall = recall[10:-10]
    # thresholds = thresholds[10:-10]

    # sample_every = max(1, len(recall) // 10)  # 最多画 200 个点
    # recall = recall[::sample_every]
    # precision = precision[::sample_every]
    # thresholds = thresholds[::sample_every]

    pr_auc = auc(recall, precision)
    pr_auc_ori = auc(recall_ori, precision_ori)

    plt.figure(figsize=(8, 6))
    # for ii, _ in enumerate(recall):
    #     # # 用箭头标注
    #     plt.annotate(
    #         f'Thr: {thresholds[ii]:.3}',
    #         xy=(recall[ii], precision[ii]),
    #         xytext=(recall[ii] - 0.1, precision[ii] + 0.1),  # 文本位置（可调）
    #         arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    #         fontsize=10,
    #     )


    plt.plot(recall, precision, marker='.', label=f'PR Curve (AUC = {pr_auc:.6f})')
    # plt.plot(recall_ori, precision_ori, marker='.', label=f'PR Curve (AUC = {pr_auc_ori:.6f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    print(f"PR AUC: {pr_auc:.4f}")


from util_heatmap import *


files = [rf"/Users/vincent/workspace/sku/matrix/new_model_matrix_dinov3.npy" ,#dinov3-vith16plus-pretrain-lvd1689m-0323-1.npy
# rf"/Users/vincent/workspace/sku/matrix/dinov3-vith16plus-pretrain-lvd1689m_similarity_jojo_data-0120-1.npy",
]
with open('/Users/vincent/workspace/sku/test_captions_dionv3.txt', 'r', encoding='utf-8') as f:
    read_data = f.read().split()
names =read_data
names = [ii.split("_")[-1] for ii in names]
# names = names[0]
# names = [ii.split("_")[-1] for ii in names]
# print(names)

idx = 0
# simi = (np.load(files[idx]) + np.load(files[-2]))/2
simi = (np.load(files[idx]))
# simi = np.load(files[-2])
if True:
    simi = np.power(simi, 2)
    # simi = simi / np.nanmax(simi)
print(f"Matrix shape: {simi.shape}")
print(f"Names length: {len(names)}")

# 确保 names 长度足够
assert len(names) >= simi.shape[0], f"Names list length ({len(names)}) is less than matrix dimension ({simi.shape[0]})"

mask = np.zeros_like(simi)
for ii in range(mask.shape[0]):
    for jj in range(mask.shape[1]):
        if ii < len(names) and jj < len(names):  # 添加边界检查
            if names[ii] == names[jj]:
                mask[ii, jj] = 1
# mask = np.zeros_like(simi)
# for ii in range(mask.shape[0]):
#     for jj in range(mask.shape[1]):
#         if names[ii] == names[jj]:
#             mask[ii, jj] = 1

print(rf"Sum of masks: {np.sum(mask)}")
compute_KL(simi, mask)
pr_curve(simi, mask)


plot_heatmap(simi, cmap='viridis', save_path=None)


if True:
    plt.figure(figsize=(8, 6))
    intersamples = (simi[mask==1])
    intrasamples = (simi[mask==0])
    # intrasamples = intrasamples[intrasamples>0.39]
    count, bins, _ = plt.hist(intersamples, bins=60, density=True, alpha=0.5, color='blue', edgecolor='k', label='Pos')
    count, bins, _ = plt.hist(intrasamples, bins=60, density=True, alpha=0.5, color='green', edgecolor='k', label='Neg')




plt.legend()
plt.show()
