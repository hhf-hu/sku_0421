import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm
import warnings
import seaborn as sns
from itertools import combinations

warnings.filterwarnings('ignore')


class AllImagesSSIMCalculator:
    def __init__(self, folder_paths, target_size=(256, 256), use_gray=True, n_workers=None):
        """
        初始化所有图片SSIM计算器

        Args:
            folder_paths: 文件夹路径列表，可以是单个文件夹或文件夹列表
            target_size: 统一调整的图片尺寸 (宽, 高)
            use_gray: 是否使用灰度图计算
            n_workers: 并行处理进程数
        """
        if isinstance(folder_paths, str):
            self.folder_paths = [folder_paths]
        else:
            self.folder_paths = folder_paths

        self.target_size = target_size
        self.use_gray = use_gray
        self.n_workers = n_workers or max(1, cpu_count() - 1)

        # 存储所有图片数据
        self.all_images = []  # 图片数据列表
        self.image_info = []  # 图片信息列表 [{filename, path, folder}]
        self.image_names = []  # 显示用名称列表
        self.total_images = 0

        # 相似度矩阵和数据框
        self.ssim_matrix = None
        self.similarity_df = None
        self.clusters = None

    def load_all_images(self):
        """加载所有文件夹的所有图片"""
        print("=" * 60)
        print("加载所有图片...")
        print("=" * 60)

        # 支持的图片格式
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}

        total_loaded = 0

        for folder_idx, folder_path in enumerate(self.folder_paths):
            folder_name = os.path.basename(folder_path)
            print(f"\n加载文件夹: {folder_name} ({folder_path})")

            if not os.path.exists(folder_path):
                print(f"警告: 文件夹不存在: {folder_path}")
                continue

            images_in_folder = 0

            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)

                # 检查是否是文件且为图片格式
                if os.path.isfile(filepath) and any(filename.lower().endswith(fmt) for fmt in supported_formats):
                    try:
                        # 读取图片
                        img = cv2.imread(filepath)
                        if img is None:
                            print(f"警告: 无法读取图片 {filename}")
                            continue

                        # 统一尺寸
                        img_resized = cv2.resize(img, self.target_size)

                        # 转换为灰度图（如果需要）
                        if self.use_gray:
                            if len(img_resized.shape) == 3:
                                img_processed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                            else:
                                img_processed = img_resized
                        else:
                            if len(img_resized.shape) == 3:
                                img_processed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                            else:
                                img_processed = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

                        # 标准化到0-1范围
                        img_processed = img_processed.astype(np.float32) / 255.0

                        # 存储图片数据
                        self.all_images.append(img_processed)

                        # 存储图片信息
                        self.image_info.append({
                            'filename': filename,
                            'path': filepath,
                            'folder': folder_name,
                            'folder_path': folder_path,
                            'folder_idx': folder_idx,
                            'global_idx': self.total_images
                        })

                        # 生成显示用名称
                        display_name = f"{folder_name}/{filename}"
                        self.image_names.append(display_name)

                        self.total_images += 1
                        images_in_folder += 1

                    except Exception as e:
                        print(f"处理图片 {filename} 时出错: {e}")

            print(f"  加载了 {images_in_folder} 张图片")
            total_loaded += images_in_folder

        print(f"\n{'=' * 60}")
        print(f"总计加载了 {total_loaded} 张图片")
        print(f"来自 {len(self.folder_paths)} 个文件夹")
        print(f"需要计算 {self.total_images * (self.total_images - 1) // 2} 对图片的SSIM")
        print(f"{'=' * 60}")

        if self.total_images < 2:
            raise ValueError("至少需要2张图片才能计算SSIM")

        return self.total_images

    def calculate_ssim_pair(self, args):
        """计算一对图片的SSIM（用于并行计算）"""
        i, j, img1, img2 = args
        try:
            if self.use_gray and len(img1.shape) == 2 and len(img2.shape) == 2:
                ssim_value = ssim(img1, img2, data_range=1.0)
            elif not self.use_gray and len(img1.shape) == 3 and len(img2.shape) == 3:
                # 对于彩色图片，计算每个通道的SSIM然后取平均
                channel_ssims = []
                for channel in range(img1.shape[2]):
                    ssim_channel = ssim(
                        img1[:, :, channel],
                        img2[:, :, channel],
                        data_range=1.0
                    )
                    channel_ssims.append(ssim_channel)
                ssim_value = np.mean(channel_ssims)
            else:
                # 如果维度不匹配，转换为灰度
                if len(img1.shape) == 3:
                    img1_gray = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    img1_gray = img1_gray.astype(np.float32) / 255.0
                else:
                    img1_gray = img1

                if len(img2.shape) == 3:
                    img2_gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    img2_gray = img2_gray.astype(np.float32) / 255.0
                else:
                    img2_gray = img2

                ssim_value = ssim(img1_gray, img2_gray, data_range=1.0)

            return (i, j, ssim_value)
        except Exception as e:
            print(f"计算图片对 ({i}, {j}) 时出错: {e}")
            return (i, j, 0.0)

    def calculate_all_pairs_ssim(self):
        """计算所有图片对之间的SSIM（包括文件夹内和跨文件夹）"""
        if self.total_images < 2:
            print("请先加载至少2张图片")
            return None

        print("\n" + "=" * 60)
        print("开始计算所有图片对的SSIM相似度...")
        print(f"使用 {self.n_workers} 个进程并行计算")
        print(f"总图片数: {self.total_images}")
        print(f"需要计算的对数: {self.total_images * (self.total_images - 1) // 2}")
        print("=" * 60)

        start_time = time.time()

        # 初始化SSIM矩阵（对称矩阵）
        n = self.total_images
        self.ssim_matrix = np.eye(n)  # 对角线设为1（自己与自己）

        # 准备所有需要计算的图片对（只计算上三角部分）
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, self.all_images[i], self.all_images[j]))

        # 并行计算
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(self.calculate_ssim_pair, pairs),
                total=len(pairs),
                desc="计算SSIM进度"
            ))

        # 填充矩阵（对称填充）
        for i, j, ssim_value in results:
            self.ssim_matrix[i, j] = ssim_value
            self.ssim_matrix[j, i] = ssim_value  # 对称性

        # 创建相似度数据框
        self._create_similarity_dataframe()

        elapsed_time = time.time() - start_time
        print(f"\nSSIM计算完成!")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"平均每对图片耗时: {elapsed_time / len(pairs) * 1000:.2f} 毫秒")

        return self.ssim_matrix

    def _create_similarity_dataframe(self):
        """创建相似度数据框（包含所有图片对）"""
        similarity_data = []

        for i in range(self.total_images):
            for j in range(i + 1, self.total_images):  # 只保存上三角部分
                info_i = self.image_info[i]
                info_j = self.image_info[j]

                similarity_data.append({
                    'Image_A': info_i['filename'],
                    'Image_B': info_j['filename'],
                    'SSIM': self.ssim_matrix[i, j],
                    'Path_A': info_i['path'],
                    'Path_B': info_j['path'],
                    'Folder_A': info_i['folder'],
                    'Folder_B': info_j['folder'],
                    'Folder_Path_A': info_i['folder_path'],
                    'Folder_Path_B': info_j['folder_path'],
                    'Same_Folder': info_i['folder'] == info_j['folder'],
                    'Type': '同一文件夹内' if info_i['folder'] == info_j['folder'] else '跨文件夹'
                })

        self.similarity_df = pd.DataFrame(similarity_data)
        # 按相似度降序排序
        self.similarity_df = self.similarity_df.sort_values('SSIM', ascending=False).reset_index(drop=True)

    def analyze_by_folder_type(self):
        """按文件夹类型分析相似度"""
        if self.similarity_df is None:
            print("请先计算SSIM")
            return None

        print("\n" + "=" * 60)
        print("按文件夹类型分析相似度")
        print("=" * 60)

        # 同一文件夹内的图片对
        same_folder_df = self.similarity_df[self.similarity_df['Same_Folder']]

        # 跨文件夹的图片对
        cross_folder_df = self.similarity_df[~self.similarity_df['Same_Folder']]

        results = {
            'all': {
                'count': len(self.similarity_df),
                'mean': self.similarity_df['SSIM'].mean(),
                'max': self.similarity_df['SSIM'].max(),
                'min': self.similarity_df['SSIM'].min()
            },
            'same_folder': {
                'count': len(same_folder_df),
                'mean': same_folder_df['SSIM'].mean() if len(same_folder_df) > 0 else 0,
                'max': same_folder_df['SSIM'].max() if len(same_folder_df) > 0 else 0,
                'min': same_folder_df['SSIM'].min() if len(same_folder_df) > 0 else 0
            },
            'cross_folder': {
                'count': len(cross_folder_df),
                'mean': cross_folder_df['SSIM'].mean() if len(cross_folder_df) > 0 else 0,
                'max': cross_folder_df['SSIM'].max() if len(cross_folder_df) > 0 else 0,
                'min': cross_folder_df['SSIM'].min() if len(cross_folder_df) > 0 else 0
            }
        }

        print(f"\n总体统计:")
        print(f"  总图片对数: {results['all']['count']}")
        print(f"  平均SSIM: {results['all']['mean']:.4f}")
        print(f"  最大SSIM: {results['all']['max']:.4f}")
        print(f"  最小SSIM: {results['all']['min']:.4f}")

        print(f"\n同一文件夹内图片对:")
        print(f"  对数: {results['same_folder']['count']}")
        print(f"  平均SSIM: {results['same_folder']['mean']:.4f}")
        print(f"  最大SSIM: {results['same_folder']['max']:.4f}")
        print(f"  最小SSIM: {results['same_folder']['min']:.4f}")

        print(f"\n跨文件夹图片对:")
        print(f"  对数: {results['cross_folder']['count']}")
        print(f"  平均SSIM: {results['cross_folder']['mean']:.4f}")
        print(f"  最大SSIM: {results['cross_folder']['max']:.4f}")
        print(f"  最小SSIM: {results['cross_folder']['min']:.4f}")

        # 可视化对比
        self._visualize_folder_type_comparison(results)

        return results

    def _visualize_folder_type_comparison(self, results):
        """可视化文件夹类型对比"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 子图1: 数量对比
        categories = ['同一文件夹内', '跨文件夹']
        counts = [results['same_folder']['count'], results['cross_folder']['count']]
        colors = ['skyblue', 'lightcoral']

        axes[0].bar(categories, counts, color=colors, edgecolor='black')
        axes[0].set_title('图片对数量对比', fontsize=14)
        axes[0].set_ylabel('对数', fontsize=12)

        # 在柱状图上添加数值
        for i, count in enumerate(counts):
            axes[0].text(i, count + max(counts) * 0.01, f'{count}',
                         ha='center', va='bottom', fontsize=11)

        # 子图2: 平均SSIM对比
        means = [results['same_folder']['mean'], results['cross_folder']['mean']]

        axes[1].bar(categories, means, color=colors, edgecolor='black')
        axes[1].set_title('平均SSIM对比', fontsize=14)
        axes[1].set_ylabel('平均SSIM', fontsize=12)
        axes[1].set_ylim(0, 1.1)

        # 在柱状图上添加数值
        for i, mean in enumerate(means):
            axes[1].text(i, mean + 0.02, f'{mean:.4f}',
                         ha='center', va='bottom', fontsize=11)

        plt.suptitle('文件夹类型对比分析', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()

    def find_most_similar_pairs(self, top_n=10, folder_type='all'):
        """找出最相似的前N对图片"""
        if self.similarity_df is None:
            print("请先计算SSIM")
            return None

        df = self.similarity_df.copy()

        if folder_type == 'same_folder':
            df = df[df['Same_Folder']]
        elif folder_type == 'cross_folder':
            df = df[~df['Same_Folder']]

        return df.head(top_n).copy()

    def find_most_dissimilar_pairs(self, top_n=10, folder_type='all'):
        """找出最不相似的前N对图片"""
        if self.similarity_df is None:
            print("请先计算SSIM")
            return None

        df = self.similarity_df.copy()

        if folder_type == 'same_folder':
            df = df[df['Same_Folder']]
        elif folder_type == 'cross_folder':
            df = df[~df['Same_Folder']]

        return df.tail(top_n).copy()

    def analyze_folder_similarity(self):
        """分析每个文件夹内部的相似度"""
        if self.similarity_df is None:
            print("请先计算SSIM")
            return None

        # 获取所有文件夹
        folders = list(set([info['folder'] for info in self.image_info]))

        print("\n" + "=" * 60)
        print("各文件夹内部相似度分析")
        print("=" * 60)

        folder_stats = {}

        for folder in folders:
            # 获取该文件夹内的所有图片对
            folder_pairs = self.similarity_df[
                (self.similarity_df['Folder_A'] == folder) &
                (self.similarity_df['Folder_B'] == folder)
                ]

            if len(folder_pairs) > 0:
                folder_stats[folder] = {
                    'image_count': len([info for info in self.image_info if info['folder'] == folder]),
                    'pair_count': len(folder_pairs),
                    'mean_ssim': folder_pairs['SSIM'].mean(),
                    'max_ssim': folder_pairs['SSIM'].max(),
                    'min_ssim': folder_pairs['SSIM'].min(),
                    'std_ssim': folder_pairs['SSIM'].std()
                }

        # 打印结果
        for folder, stats in folder_stats.items():
            print(f"\n文件夹: {folder}")
            print(f"  图片数量: {stats['image_count']}")
            print(f"  内部比较对数: {stats['pair_count']}")
            print(f"  平均SSIM: {stats['mean_ssim']:.4f}")
            print(f"  最大SSIM: {stats['max_ssim']:.4f}")
            print(f"  最小SSIM: {stats['min_ssim']:.4f}")
            print(f"  SSIM标准差: {stats['std_ssim']:.4f}")

        # 可视化
        self._visualize_folder_similarity(folder_stats)

        return folder_stats

    def _visualize_folder_similarity(self, folder_stats):
        """可视化各文件夹内部相似度"""
        if not folder_stats:
            return

        folders = list(folder_stats.keys())
        mean_ssims = [folder_stats[f]['mean_ssim'] for f in folders]
        image_counts = [folder_stats[f]['image_count'] for f in folders]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 子图1: 平均SSIM柱状图
        bars1 = axes[0].bar(folders, mean_ssims, color='lightgreen', edgecolor='black')
        axes[0].set_title('各文件夹内部平均SSIM', fontsize=14)
        axes[0].set_xlabel('文件夹', fontsize=12)
        axes[0].set_ylabel('平均SSIM', fontsize=12)
        axes[0].set_ylim(0, 1.1)
        axes[0].tick_params(axis='x', rotation=45)

        # 在柱状图上添加数值
        for bar, value in zip(bars1, mean_ssims):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                         f'{value:.4f}', ha='center', va='bottom', fontsize=10)

        # 子图2: 图片数量与平均SSIM散点图
        scatter = axes[1].scatter(image_counts, mean_ssims, s=100, alpha=0.6, edgecolors='black')
        axes[1].set_title('图片数量 vs 平均SSIM', fontsize=14)
        axes[1].set_xlabel('图片数量', fontsize=12)
        axes[1].set_ylabel('平均SSIM', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # 添加标签
        for i, folder in enumerate(folders):
            axes[1].annotate(folder, (image_counts[i], mean_ssims[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)

        plt.suptitle('文件夹内部相似度分析', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()

    def visualize_ssim_matrix(self, figsize=(14, 12), cmap='viridis'):
        """可视化完整的SSIM矩阵"""
        if self.ssim_matrix is None:
            print("请先计算SSIM")
            return

        plt.figure(figsize=figsize)

        # 创建热力图
        im = plt.imshow(self.ssim_matrix, cmap=cmap, aspect='auto')
        plt.colorbar(im, fraction=0.046, pad=0.04)


        # 获取文件夹分界线
        folder_boundaries = []
        current_folder = None

        for i, info in enumerate(self.image_info):
            if info['folder'] != current_folder:
                folder_boundaries.append(i)
                current_folder = info['folder']

        # 添加文件夹分界线
        for boundary in folder_boundaries[1:]:
            plt.axhline(y=boundary - 0.5, color='white', linewidth=2, linestyle='--')
            plt.axvline(x=boundary - 0.5, color='white', linewidth=2, linestyle='--')

        # 设置坐标轴标签
        plt.xticks(range(len(self.image_names)), self.image_names, rotation=90, fontsize=8)
        plt.yticks(range(len(self.image_names)), self.image_names, fontsize=8)

        plt.title(f'SSIM similarity matrix ({self.total_images}images)', fontsize=16, pad=20)
        plt.xlabel('Image_Category', fontsize=12)
        plt.ylabel('Image_Category', fontsize=12)

        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join('full_ssim_matrix_heatmap.png'),
                    dpi=150, bbox_inches='tight')

    def visualize_similarity_distribution(self, bins=20):
        """可视化相似度分布"""
        if self.similarity_df is None:
            print("请先计算SSIM")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 子图1: 所有图片对的SSIM分布直方图
        axes[0, 0].hist(self.similarity_df['SSIM'], bins=bins, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('SSIM值')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('所有图片对SSIM分布')
        axes[0, 0].grid(True, alpha=0.3)

        # 子图2: 箱线图
        data_to_plot = []
        labels = []

        # 所有图片对
        data_to_plot.append(self.similarity_df['SSIM'])
        labels.append('所有图片')

        # 同一文件夹内
        same_folder_df = self.similarity_df[self.similarity_df['Same_Folder']]
        if len(same_folder_df) > 0:
            data_to_plot.append(same_folder_df['SSIM'])
            labels.append('同一文件夹')

        # 跨文件夹
        cross_folder_df = self.similarity_df[~self.similarity_df['Same_Folder']]
        if len(cross_folder_df) > 0:
            data_to_plot.append(cross_folder_df['SSIM'])
            labels.append('跨文件夹')

        axes[0, 1].boxplot(data_to_plot, labels=labels, vert=True)
        axes[0, 1].set_ylabel('SSIM值')
        axes[0, 1].set_title('SSIM分布箱线图对比')
        axes[0, 1].grid(True, alpha=0.3)

        # 子图3: 密度图
        if len(same_folder_df) > 0:
            sns.kdeplot(same_folder_df['SSIM'], ax=axes[1, 0], label='同一文件夹', linewidth=2)
        if len(cross_folder_df) > 0:
            sns.kdeplot(cross_folder_df['SSIM'], ax=axes[1, 0], label='跨文件夹', linewidth=2)

        axes[1, 0].set_xlabel('SSIM值')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].set_title('SSIM密度分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 子图4: 累积分布图
        sorted_ssim = np.sort(self.similarity_df['SSIM'])
        cdf = np.arange(1, len(sorted_ssim) + 1) / len(sorted_ssim)

        axes[1, 1].plot(sorted_ssim, cdf, linewidth=2)
        axes[1, 1].set_xlabel('SSIM值')
        axes[1, 1].set_ylabel('累积概率')
        axes[1, 1].set_title('SSIM累积分布函数')
        axes[1, 1].grid(True, alpha=0.3)

        # 添加关键百分位标记
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for p in percentiles:
            value = np.percentile(sorted_ssim, p * 100)
            axes[1, 1].axvline(x=value, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].text(value, 0.5, f'{p * 100:.0f}%', rotation=90,
                            verticalalignment='center', fontsize=9)

        plt.suptitle('SSIM相似度分布分析', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    def cluster_images(self, threshold=0.7):
        """基于SSIM阈值进行图片聚类"""
        if self.ssim_matrix is None:
            print("请先计算SSIM")
            return None

        n = self.total_images
        visited = [False] * n
        clusters = []

        for i in range(n):
            if not visited[i]:
                # 创建新聚类
                cluster = [i]
                visited[i] = True

                # 查找相似图片
                for j in range(n):
                    if not visited[j] and self.ssim_matrix[i, j] >= threshold:
                        cluster.append(j)
                        visited[j] = True

                clusters.append(cluster)

        # 转换为图片信息的聚类
        self.clusters = []
        for cluster_indices in clusters:
            cluster_info = []
            for idx in cluster_indices:
                info = self.image_info[idx].copy()
                info['global_idx'] = idx
                cluster_info.append(info)
            self.clusters.append(cluster_info)

        print(f"\n基于SSIM ≥ {threshold} 的聚类结果:")
        print(f"共找到 {len(self.clusters)} 个聚类")

        for i, cluster in enumerate(self.clusters):
            print(f"\n聚类 {i + 1} ({len(cluster)}张图片):")
            folders_in_cluster = {}
            for info in cluster:
                folder = info['folder']
                folders_in_cluster[folder] = folders_in_cluster.get(folder, 0) + 1

            for folder, count in folders_in_cluster.items():
                print(f"  {folder}: {count}张")

        return self.clusters

    def find_similar_images(self, image_name, threshold=0.8):
        """找出与指定图片相似的所有图片"""
        if self.ssim_matrix is None:
            print("请先计算SSIM")
            return None

        # 查找图片索引
        target_idx = -1
        for i, info in enumerate(self.image_info):
            if info['filename'] == image_name:
                target_idx = i
                break

        if target_idx == -1:
            print(f"未找到图片: {image_name}")
            return None

        similar_images = []

        for i, info in enumerate(self.image_info):
            if i != target_idx:  # 排除自己
                similarity = self.ssim_matrix[target_idx, i]
                if similarity >= threshold:
                    similar_images.append({
                        'Image': info['filename'],
                        'Folder': info['folder'],
                        'SSIM': similarity,
                        'Path': info['path'],
                        'Same_Folder': info['folder'] == self.image_info[target_idx]['folder']
                    })

        # 按相似度降序排序
        similar_images.sort(key=lambda x: x['SSIM'], reverse=True)

        return pd.DataFrame(similar_images)

    def export_results(self, output_dir='./all_images_ssim_results'):
        """导出所有计算结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 导出相似度表格
        csv_path = os.path.join(output_dir, 'all_images_similarity.csv')
        self.similarity_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"相似度表格已保存到: {csv_path}")

        # 导出SSIM矩阵
        matrix_path = os.path.join(output_dir, 'full_ssim_matrix.npy')
        np.save(matrix_path, self.ssim_matrix)

        # 导出图片信息
        info_df = pd.DataFrame(self.image_info)
        info_path = os.path.join(output_dir, 'all_images_info.csv')
        info_df.to_csv(info_path, index=False, encoding='utf-8-sig')

        # 按文件夹类型分别导出
        same_folder_df = self.similarity_df[self.similarity_df['Same_Folder']]
        cross_folder_df = self.similarity_df[~self.similarity_df['Same_Folder']]

        same_folder_path = os.path.join(output_dir, 'same_folder_similarity.csv')
        cross_folder_path = os.path.join(output_dir, 'cross_folder_similarity.csv')

        same_folder_df.to_csv(same_folder_path, index=False, encoding='utf-8-sig')
        cross_folder_df.to_csv(cross_folder_path, index=False, encoding='utf-8-sig')

        # 生成详细报告
        self._generate_detailed_report(output_dir)

        # 保存可视化图表
        self._save_visualizations(output_dir)

        print(f"\n所有结果已保存到目录: {output_dir}")

    def _generate_detailed_report(self, output_dir):
        """生成详细分析报告"""
        report_path = os.path.join(output_dir, 'detailed_analysis_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("所有图片SSIM相似度分析详细报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析文件夹: {', '.join([os.path.basename(p) for p in self.folder_paths])}\n")
            f.write(f"总图片数量: {self.total_images}\n")
            f.write(f"总图片对数: {len(self.similarity_df)}\n\n")

            # 总体统计
            f.write("=" * 80 + "\n")
            f.write("总体统计\n")
            f.write("=" * 80 + "\n")
            f.write(f"平均SSIM: {self.similarity_df['SSIM'].mean():.4f}\n")
            f.write(f"SSIM中位数: {self.similarity_df['SSIM'].median():.4f}\n")
            f.write(f"SSIM最大值: {self.similarity_df['SSIM'].max():.4f}\n")
            f.write(f"SSIM最小值: {self.similarity_df['SSIM'].min():.4f}\n")
            f.write(f"SSIM标准差: {self.similarity_df['SSIM'].std():.4f}\n\n")

            # 文件夹类型分析
            same_folder_df = self.similarity_df[self.similarity_df['Same_Folder']]
            cross_folder_df = self.similarity_df[~self.similarity_df['Same_Folder']]

            f.write("=" * 80 + "\n")
            f.write("文件夹类型分析\n")
            f.write("=" * 80 + "\n")
            f.write(f"同一文件夹内图片对数: {len(same_folder_df)}\n")
            f.write(f"同一文件夹内平均SSIM: {same_folder_df['SSIM'].mean():.4f}\n")
            f.write(f"跨文件夹图片对数: {len(cross_folder_df)}\n")
            f.write(f"跨文件夹平均SSIM: {cross_folder_df['SSIM'].mean():.4f}\n\n")

            # 最相似的图片对
            f.write("=" * 80 + "\n")
            f.write("最相似的20对图片\n")
            f.write("=" * 80 + "\n")
            top20 = self.find_most_similar_pairs(20)
            for _, row in top20.iterrows():
                folder_type = "同一文件夹" if row['Same_Folder'] else "跨文件夹"
                f.write(f"{row['Image_A']:25s} <-> {row['Image_B']:25s} : "
                        f"SSIM = {row['SSIM']:.4f} ({folder_type})\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("最不相似的20对图片\n")
            f.write("=" * 80 + "\n")
            bottom20 = self.find_most_dissimilar_pairs(20)
            for _, row in bottom20.iterrows():
                folder_type = "同一文件夹" if row['Same_Folder'] else "跨文件夹"
                f.write(f"{row['Image_A']:25s} <-> {row['Image_B']:25s} : "
                        f"SSIM = {row['SSIM']:.4f} ({folder_type})\n")

            # 文件夹内部分析
            f.write("\n" + "=" * 80 + "\n")
            f.write("各文件夹内部相似度分析\n")
            f.write("=" * 80 + "\n")

            folders = list(set([info['folder'] for info in self.image_info]))
            for folder in folders:
                folder_pairs = same_folder_df[
                    (same_folder_df['Folder_A'] == folder) &
                    (same_folder_df['Folder_B'] == folder)
                    ]

                if len(folder_pairs) > 0:
                    image_count = len([info for info in self.image_info if info['folder'] == folder])
                    f.write(f"\n文件夹: {folder}\n")
                    f.write(f"  图片数量: {image_count}\n")
                    f.write(f"  内部比较对数: {len(folder_pairs)}\n")
                    f.write(f"  平均SSIM: {folder_pairs['SSIM'].mean():.4f}\n")
                    f.write(f"  最大SSIM: {folder_pairs['SSIM'].max():.4f}\n")
                    f.write(f"  最小SSIM: {folder_pairs['SSIM'].min():.4f}\n")

            # 相似度分布统计
            f.write("\n" + "=" * 80 + "\n")
            f.write("相似度分布统计\n")
            f.write("=" * 80 + "\n")

            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            hist, bin_edges = np.histogram(self.similarity_df['SSIM'], bins=bins)

            for i in range(len(hist)):
                lower = bin_edges[i]
                upper = bin_edges[i + 1]
                count = hist[i]
                percentage = count / len(self.similarity_df) * 100
                f.write(f"  {lower:.1f} ≤ SSIM < {upper:.1f}: {count:6d} 对 ({percentage:6.2f}%)\n")

        print(f"详细分析报告已保存到: {report_path}")

    def _save_visualizations(self, output_dir):
        """保存可视化图表"""
        # 保存SSIM矩阵热力图
        plt.figure(figsize=(14, 12))
        plt.imshow(self.ssim_matrix, cmap='viridis')
        plt.colorbar()
        plt.title(f'SSIM_similarity_matrix ({self.total_images}images)')
        plt.xlabel('Image_Category')
        plt.ylabel('Image_Category')
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(output_dir, 'full_ssim_matrix.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 保存分布图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 直方图
        axes[0].hist(self.similarity_df['SSIM'], bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('SSIM值')
        axes[0].set_ylabel('频数')
        axes[0].set_title('SSIM分布直方图')
        axes[0].grid(True, alpha=0.3)

        # 箱线图
        same_folder_df = self.similarity_df[self.similarity_df['Same_Folder']]
        cross_folder_df = self.similarity_df[~self.similarity_df['Same_Folder']]

        data_to_plot = [self.similarity_df['SSIM']]
        labels = ['所有图片']

        if len(same_folder_df) > 0:
            data_to_plot.append(same_folder_df['SSIM'])
            labels.append('同一文件夹')

        if len(cross_folder_df) > 0:
            data_to_plot.append(cross_folder_df['SSIM'])
            labels.append('跨文件夹')

        axes[1].boxplot(data_to_plot, labels=labels, vert=True)
        axes[1].set_ylabel('SSIM值')
        axes[1].set_title('SSIM分布箱线图')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('SSIM相似度分布', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ssim_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def main():
    # 设置你的文件夹路径
    folder_paths = [
        "/Users/vincent/workspace/sku/Unit0001",
        "/Users/vincent/workspace/sku/Unit0002",
        "/Users/vincent/workspace/sku/Unit0003"
        # 可以添加更多文件夹
        # "/path/to/folder3",
        # "/path/to/folder4"
    ]

    print("=" * 80)
    print("所有图片SSIM相似度分析（包括文件夹内和跨文件夹）")
    print("=" * 80)

    try:
        # 创建计算器实例
        calculator = AllImagesSSIMCalculator(
            folder_paths=folder_paths,
            target_size=(256, 256),  # 统一尺寸
            use_gray=True,  # 使用灰度图计算（更快）
            n_workers=4  # 并行计算进程数
        )

        # 1. 加载所有图片
        total_images = calculator.load_all_images()

        # 2. 计算所有图片对的SSIM
        ssim_matrix = calculator.calculate_all_pairs_ssim()

        if ssim_matrix is not None:
            # 3. 按文件夹类型分析
            calculator.analyze_by_folder_type()

            # 4. 各文件夹内部相似度分析
            calculator.analyze_folder_similarity()

            # 5. 显示最相似的图片对
            print("\n" + "=" * 80)
            print("最相似的15对图片（所有图片中）:")
            print("=" * 80)
            top_pairs = calculator.find_most_similar_pairs(15)
            for idx, (_, row) in enumerate(top_pairs.iterrows(), 1):
                folder_type = "同一文件夹" if row['Same_Folder'] else "跨文件夹"
                print(f"{idx:3d}. {row['Image_A']:25s} <-> {row['Image_B']:25s} : "
                      f"SSIM = {row['SSIM']:.4f} ({folder_type})")

            # 6. 显示同一文件夹内最相似的图片对
            print("\n" + "=" * 80)
            print("同一文件夹内最相似的10对图片:")
            print("=" * 80)
            same_folder_top = calculator.find_most_similar_pairs(10, 'same_folder')
            for idx, (_, row) in enumerate(same_folder_top.iterrows(), 1):
                print(f"{idx:3d}. {row['Image_A']:25s} <-> {row['Image_B']:25s} : "
                      f"SSIM = {row['SSIM']:.4f} ({row['Folder_A']})")

            # 7. 显示跨文件夹最相似的图片对
            print("\n" + "=" * 80)
            print("跨文件夹最相似的10对图片:")
            print("=" * 80)
            cross_folder_top = calculator.find_most_similar_pairs(10, 'cross_folder')
            for idx, (_, row) in enumerate(cross_folder_top.iterrows(), 1):
                print(f"{idx:3d}. {row['Image_A']:25s} <-> {row['Image_B']:25s} : "
                      f"SSIM = {row['SSIM']:.4f} ({row['Folder_A']} ↔ {row['Folder_B']})")

            # 8. 相似度分布可视化
            calculator.visualize_similarity_distribution()

            # 9. 完整SSIM矩阵可视化（如果图片数量不太多）
            if total_images <= 50:
                calculator.visualize_ssim_matrix()
            else:
                print(f"\n图片数量较多 ({total_images}张)，不显示完整矩阵热力图")
                print("可在导出结果中查看矩阵数据")

            # 10. 聚类分析
            calculator.cluster_images(threshold=0.7)

            # 11. 导出所有结果
            calculator.export_results('./complete_ssim_analysis')

            # 12. 查询示例
            print("\n" + "=" * 80)
            print("查询示例:")
            print("=" * 80)

            # 示例：查询与某张图片相似的所有图片
            if calculator.image_info:
                sample_image = calculator.image_info[0]['filename']
                similar_images = calculator.find_similar_images(sample_image, threshold=0.6)

                if similar_images is not None and len(similar_images) > 0:
                    print(f"\n与 '{sample_image}' 相似的图片 (SSIM ≥ 0.6):")
                    for _, row in similar_images.head(10).iterrows():
                        folder_type = "同一文件夹" if row['Same_Folder'] else "其他文件夹"
                        print(f"  {row['Image']} ({row['Folder']}): SSIM = {row['SSIM']:.4f} ({folder_type})")
                else:
                    print(f"\n没有找到与 '{sample_image}' 相似的图片 (SSIM ≥ 0.6)")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查:")
        print("1. 文件夹路径是否正确")
        print("2. 文件夹中是否有图片文件")
        print("3. 是否安装了必要的库: pip install opencv-python numpy pandas matplotlib scikit-image tqdm seaborn")


# 快速使用函数
def quick_all_images_analysis(folder_paths, output_dir='./all_images_results'):
    """
    快速分析所有图片的SSIM相似度
    """
    start_time = time.time()

    try:
        calculator = AllImagesSSIMCalculator(folder_paths)
        calculator.load_all_images()

        if calculator.total_images < 2:
            print(f"图片数量不足: {calculator.total_images}")
            return None

        calculator.calculate_all_pairs_ssim()
        calculator.export_results(output_dir)

        elapsed_time = time.time() - start_time
        print(f"\n分析完成!")
        print(f"总图片数: {calculator.total_images}")
        print(f"总图片对数: {calculator.total_images * (calculator.total_images - 1) // 2}")
        print(f"总耗时: {elapsed_time:.2f} 秒")

        return calculator

    except Exception as e:
        print(f"分析出错: {e}")
        return None


if __name__ == "__main__":
    # 使用你的具体路径
    folders = [
        "/Users/vincent/workspace/sku/Unit0001",
        "/Users/vincent/workspace/sku/Unit0002"
        # "/Users/vincent/workspace/sku/Unit0003"
    ]

    # 方式1: 完整分析
    main()

    # 方式2: 快速分析
    # calculator = quick_all_images_analysis(folders)