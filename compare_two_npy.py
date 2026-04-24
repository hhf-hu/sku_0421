"""
compare_two_npy.py
------------------
对两个 .npy 相似度矩阵文件计算各项指标，并将 PR 曲线画在同一张图中。

使用方法：
  1. 修改下方 FILE_A / FILE_B 为实际路径
  2. 根据需要调整 LABEL_A / LABEL_B（图例名称）
  3. 运行: python compare_two_npy.py
"""

import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False
# ===================== 配置区 =====================
FILE_A = r"/sku/matrix/old_model_matrix.npy"
FILE_B = r"/sku/matrix/new_model_matrix.npy"

LABEL_A = "old"
LABEL_B = "new"

# 是否对相似度矩阵做 power + normalize（与原代码一致）
APPLY_POWER_NORM = True

# 阈值列表，用于计算 TP/FP 统计
THRESHOLDS_TO_REPORT = [0.5, 0.6, 0.7]

# ===================== 标签（names）=====================

with open('/Users/vincent/workspace/sku/test_captions.txt', 'r', encoding='utf-8') as f:
    read_data = f.read().split()
names =read_data
names = [ii.split("_")[-1] for ii in names]


# ===================== 核心函数 =====================

def load_and_preprocess(path, apply_power_norm=True):
    simi = np.load(path)
    if apply_power_norm:
        simi = np.power(simi, 2)
        simi = simi / np.nanmax(simi)
    return simi


def build_mask(simi, names):
    """mask[i,j]=1 表示同类（正对），=0 表示不同类（负对），仅取上三角"""
    mask = np.zeros_like(simi, dtype=int)
    for ii in range(mask.shape[0]):
        for jj in range(mask.shape[1]):
            if ii < len(names) and jj < len(names):
                if names[ii] == names[jj]:
                    mask[ii, jj] = 1
    # 屏蔽下三角（含对角）
    for ii in range(mask.shape[0]):
        for jj in range(mask.shape[1]):
            if ii >= jj:
                mask[ii, jj] = 3   # 3 = ignore
    return mask


def compute_metrics(simi, mask, label="", thresholds=None):
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7]

    pos_scores = list(simi[mask == 1])
    neg_scores = list(simi[mask == 0])

    print(f"\n{'='*50}")
    print(f"[{label}]")
    print(f"  矩阵大小: {simi.shape}")
    print(f"  正对数量 (上三角): {len(pos_scores)}")
    print(f"  负对数量 (上三角): {len(neg_scores)}")

    # ---------- KL / JS ----------
    bins = np.linspace(
        min(np.min(pos_scores), np.min(neg_scores)),
        max(np.max(pos_scores), np.max(neg_scores)),
        61
    )
    count_pos, _ = np.histogram(pos_scores, bins=bins, density=True)
    count_neg, _ = np.histogram(neg_scores, bins=bins, density=True)
    eps = 1e-10
    p = np.clip(count_pos, eps, None)
    q = np.clip(count_neg, eps, None)
    bin_width = np.diff(bins)[0]
    kl_pq = np.sum(p * np.log(p / q)) * bin_width
    kl_qp = np.sum(q * np.log(q / p)) * bin_width
    js = 0.5 * (kl_pq + kl_qp)
    print(f"  KL(Pos||Neg): {kl_pq:.4f}")
    print(f"  KL(Neg||Pos): {kl_qp:.4f}")
    print(f"  JS Divergence: {js:.4f}")

    # ---------- 阈值统计 ----------
    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)
    for thr in thresholds:
        tp = np.sum(pos_arr > thr)
        fp = np.sum(neg_arr > thr)
        print(f"  Threshold={thr}: TP={tp}/{len(pos_arr)} ({100*tp/len(pos_arr):.1f}%),  "
              f"FP={fp}/{len(neg_arr)} ({100*fp/len(neg_arr):.1f}%)")

    # ---------- PR-AUC (原始不平衡) ----------
    y_true_ori = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores_ori = pos_scores + neg_scores
    prec_ori, rec_ori, _ = precision_recall_curve(y_true_ori, y_scores_ori)
    pr_auc_ori = auc(rec_ori, prec_ori)
    print(f"  PR-AUC (原始): {pr_auc_ori:.6f}")

    # ---------- PR-AUC (平衡采样) ----------
    neg_balanced = list(np.random.choice(neg_scores, size=len(pos_scores), replace=len(neg_scores) < len(pos_scores)))
    y_true_bal = [1] * len(pos_scores) + [0] * len(neg_balanced)
    y_scores_bal = pos_scores + neg_balanced
    prec_bal, rec_bal, _ = precision_recall_curve(y_true_bal, y_scores_bal)
    pr_auc_bal = auc(rec_bal, prec_bal)
    print(f"  PR-AUC (平衡): {pr_auc_bal:.6f}")

    # ---------- ROC-AUC ----------
    roc_auc = roc_auc_score(y_true_ori, y_scores_ori)
    print(f"  ROC-AUC: {roc_auc:.6f}")

    return {
        "kl_pq": kl_pq, "kl_qp": kl_qp, "js": js,
        "pr_auc_ori": pr_auc_ori, "pr_auc_bal": pr_auc_bal,
        "roc_auc": roc_auc,
        "prec_ori": prec_ori, "rec_ori": rec_ori,
        "prec_bal": prec_bal, "rec_bal": rec_bal,
        "pos_scores": pos_scores, "neg_scores": neg_scores,
    }


def plot_pr_curves(results_a, results_b, label_a, label_b, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # ---------- 左图：原始不平衡 PR ----------
    # ax = axes[0]
    # ax.plot(results_a["rec_ori"], results_a["prec_ori"], label=f"{label_a}  AUC={results_a['pr_auc_ori']:.4f}", linewidth=1.5)
    # ax.plot(results_b["rec_ori"], results_b["prec_ori"], label=f"{label_b}  AUC={results_b['pr_auc_ori']:.4f}", linewidth=1.5)
    # ax.set_xlabel("Recall")
    # ax.set_ylabel("Precision")
    # ax.set_title("PR Curve（原始不平衡）")
    # ax.legend()
    # ax.grid(True)
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1.05])

    # ---------- 右图：平衡采样 PR ----------
    # ax = axes[1]
    ax.plot(results_a["rec_bal"], results_a["prec_bal"], label=f"{label_a}  AUC={results_a['pr_auc_bal']:.4f}", linewidth=1.5)
    ax.plot(results_b["rec_bal"], results_b["prec_bal"], label=f"{label_b}  AUC={results_b['pr_auc_bal']:.4f}", linewidth=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve（平衡采样）")
    ax.legend()
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[已保存 PR 曲线图] -> {save_path}")
    plt.show()


def plot_score_distributions(results_a, results_b, label_a, label_b, save_path=None):
    """正负对分数分布对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res, label in zip(axes, [results_a, results_b], [label_a, label_b]):
        ax.hist(res["pos_scores"], bins=60, density=True, alpha=0.5, color="blue", label="Pos")
        ax.hist(res["neg_scores"], bins=60, density=True, alpha=0.5, color="green", label="Neg")
        ax.set_title(f"Score Distribution — {label}")
        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[已保存分布图] -> {save_path}")
    plt.show()


# ===================== 主流程 =====================
if __name__ == "__main__":
    np.random.seed(42)

    print(f"加载文件 A: {FILE_A}")
    simi_a = load_and_preprocess(FILE_A, APPLY_POWER_NORM)
    print(f"加载文件 B: {FILE_B}")
    simi_b = load_and_preprocess(FILE_B, APPLY_POWER_NORM)

    assert simi_a.shape == simi_b.shape, \
        f"两个矩阵形状不一致！A={simi_a.shape}, B={simi_b.shape}"
    assert simi_a.shape[0] <= len(names), \
        f"names 长度({len(names)}) < 矩阵维度({simi_a.shape[0]})"

    # 构建 mask（两个文件共用）
    mask = build_mask(simi_a, names)

    # 计算指标
    results_a = compute_metrics(simi_a, mask, label=LABEL_A, thresholds=THRESHOLDS_TO_REPORT)
    results_b = compute_metrics(simi_b, mask, label=LABEL_B, thresholds=THRESHOLDS_TO_REPORT)

    # 汇总对比
    print(f"\n{'='*50}")
    print("【指标对比汇总】")
    header = f"{'指标':<20} {LABEL_A:>20} {LABEL_B:>20}"
    print(header)
    print("-" * len(header))
    metrics_to_compare = [
        ("KL(Pos||Neg)", "kl_pq"),
        ("KL(Neg||Pos)", "kl_qp"),
        ("JS Divergence", "js"),
        ("PR-AUC (原始)", "pr_auc_ori"),
        ("PR-AUC (平衡)", "pr_auc_bal"),
        ("ROC-AUC", "roc_auc"),
    ]
    for name, key in metrics_to_compare:
        va = results_a[key]
        vb = results_b[key]
        better = "← A更好" if va > vb else ("← B更好" if vb > va else "持平")
        if key in ("kl_pq", "kl_qp", "js"):  # 越大分离越好
            better = "← A更好" if va > vb else ("← B更好" if vb > va else "持平")
        print(f"{name:<20} {va:>20.6f} {vb:>20.6f}  {better}")

    # 画图
    plot_pr_curves(results_a, results_b, LABEL_A, LABEL_B,
                   save_path="pr_curves_comparison.png")
    plot_score_distributions(results_a, results_b, LABEL_A, LABEL_B,
                             save_path="score_distributions.png")
