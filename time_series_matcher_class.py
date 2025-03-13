import numpy as np
from scipy import stats
from tslearn.metrics import dtw
from fastdtw import fastdtw
from typing import List, Tuple, Union, Dict
import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
import platform
import os
import time
from scipy.signal import decimate

# 配置中文字体
def configure_chinese_font():
    """配置中文字体，返回字体属性"""
    system = platform.system()
    if system == 'Windows':
        font_paths = [
            r"C:\Windows\Fonts\SimHei.ttf",  # 黑体
            r"C:\Windows\Fonts\msyh.ttc",    # 微软雅黑
            r"C:\Windows\Fonts\simsun.ttc"   # 宋体
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc'
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        ]
    
    # 尝试设置字体
    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = FontProperties(fname=font_path)
                plt.rcParams['font.sans-serif'] = [font.get_name()]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"成功加载中文字体: {font_path}")
                break
            except Exception as e:
                print(f"加载字体 {font_path} 失败: {str(e)}")
    
    if font is None:
        print("警告：未能找到合适的中文字体，将使用系统默认字体")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        font = FontProperties()
    
    return font

# 初始化中文字体
chinese_font = configure_chinese_font()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TimeSeriesMatcher:
    def __init__(self, context_sequence: np.ndarray, threshold: float = 0.5, 
                 radius: int = 1, position_group_ratio: float = 0.1, 
                 lb_keogh_multiplier: float = 1.5, downsample_factor: int = 2):
        """
        初始化时间序列匹配器
        
        Args:
            context_sequence: 形状为 (length, dimensions) 的上下文序列
            threshold: DTW距离阈值，用于确定匹配
            radius: DTW warping半径，影响DTW计算的弹性程度
            position_group_ratio: 用于确定相邻匹配分组的比例，值越小分组越严格
            lb_keogh_multiplier: LB_Keogh剪枝倍数，值越小剪枝越严格
            downsample_factor: 降采样因子，默认为2（每2个点取1个点）
        """
        self.context_sequence = context_sequence
        self.threshold = threshold
        self.radius = radius
        self.position_group_ratio = position_group_ratio
        self.lb_keogh_multiplier = lb_keogh_multiplier
        self.downsample_factor = downsample_factor
        
    def _downsample_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        对序列进行降采样
        
        Args:
            sequence: 输入序列，形状为 (length, dimensions)
            
        Returns:
            np.ndarray: 降采样后的序列
        """
        if self.downsample_factor <= 1:
            return sequence
            
        downsampled = np.zeros((len(sequence) // self.downsample_factor + 
                               (1 if len(sequence) % self.downsample_factor != 0 else 0),
                               sequence.shape[1]))
        
        for dim in range(sequence.shape[1]):
            # 使用scipy的decimate函数进行降采样，它会自动应用低通滤波器
            downsampled[:, dim] = decimate(sequence[:, dim], 
                                         self.downsample_factor, 
                                         ftype='fir',
                                         zero_phase=True)
        
        return downsampled
    
    def _compute_lb_keogh(self, query: np.ndarray, window: np.ndarray) -> float:
        """
        计算LB_Keogh下界，使用早期放弃策略
        
        Args:
            query: 查询序列
            window: 待比较的窗口
            
        Returns:
            float: LB_Keogh下界值，如果超过阈值则返回float('inf')
        """
        length = len(query)
        n_dims = query.shape[1]
        
        # 计算早期放弃阈值
        early_abandon_threshold = (self.threshold * self.lb_keogh_multiplier) ** 2 * length
        running_sum = 0
        
        # 使用向量化操作计算上下包络
        r = self.radius
        upper_env = np.zeros_like(query)
        lower_env = np.zeros_like(query)
        
        # 一次性计算所有维度的上下包络
        for i in range(length):
            start = max(0, i - r)
            end = min(length, i + r + 1)
            upper_env[i] = np.max(window[start:end], axis=0)
            lower_env[i] = np.min(window[start:end], axis=0)
            
            # 计算当前点的距离并更新running_sum
            point_dist = np.sum(np.where(query[i] > upper_env[i],
                                      (query[i] - upper_env[i]) ** 2,
                                      np.where(query[i] < lower_env[i],
                                             (query[i] - lower_env[i]) ** 2,
                                             0)))
            running_sum += point_dist
            
            # 早期放弃检查
            if running_sum > early_abandon_threshold:
                return float('inf')
        
        return np.sqrt(running_sum / length)
    
    def find_matches(self, query: np.ndarray) -> Tuple[List[Tuple[int, float]], Dict[str, float]]:
        """
        在上下文序列中查找与查询序列相似的子序列
        
        Args:
            query: 形状为 (length, dimensions) 的查询序列
            
        Returns:
            Tuple[List[Tuple[int, float]], Dict[str, float]]: 
                - 匹配结果列表，每个元素为 (起始位置, DTW距离)
                - 时间统计字典，包含各阶段耗时
        """
        time_stats = {
            "preprocessing": 0,
            "lb_keogh_pruning": 0,
            "dtw_calculation": 0,
            "post_processing": 0,
            "total": 0
        }
        
        total_start_time = time.time()
        
        # 预处理：降采样
        preprocess_start = time.time()
        query_downsampled = self._downsample_sequence(query)
        context_downsampled = self._downsample_sequence(self.context_sequence)
        time_stats["preprocessing"] = time.time() - preprocess_start
        
        query_length = len(query_downsampled)
        matches = []
        
        # 滑动窗口搜索
        logger.info(f"开始查找匹配，查询序列长度: {query_length}，阈值: {self.threshold}")
        
        lb_keogh_total = 0
        dtw_total = 0
        
        for i in range(len(context_downsampled) - query_length + 1):
            window = context_downsampled[i:i+query_length]
            
            # 使用优化后的LB_Keogh进行快速剪枝
            lb_start = time.time()
            lb_dist = self._compute_lb_keogh(query_downsampled, window)
            lb_time = time.time() - lb_start
            lb_keogh_total += lb_time
            
            if lb_dist == float('inf'):
                continue
                
            if lb_dist > self.threshold * self.lb_keogh_multiplier:
                logger.debug(f"位置 {i}: LB_Keogh距离 {lb_dist:.4f} > {self.threshold * self.lb_keogh_multiplier:.4f}，剪枝跳过")
                continue
            
            # 对通过剪枝的窗口计算DTW距离
            dtw_start = time.time()
            distance, path = fastdtw(query_downsampled, window, radius=self.radius)
            dtw_time = time.time() - dtw_start
            dtw_total += dtw_time
            
            # 计算归一化DTW距离
            path_length = len(path)
            distance = np.sqrt(distance / path_length)
            
            # 如果找到匹配，将原始（未降采样）位置添加到结果中
            if distance <= self.threshold:
                original_pos = i * self.downsample_factor
                matches.append((original_pos, distance))
                logger.info(f"找到匹配: 位置 {original_pos}, DTW距离: {distance:.4f}")
        
        time_stats["lb_keogh_pruning"] = lb_keogh_total
        time_stats["dtw_calculation"] = dtw_total
        
        # 后处理：合并相近的匹配
        post_start = time.time()
        merged_matches = self._merge_matches(matches, len(query))
        time_stats["post_processing"] = time.time() - post_start
        time_stats["total"] = time.time() - total_start_time
        
        return merged_matches, time_stats
        
    def _merge_matches(self, matches: List[Tuple[int, float]], query_length: int) -> List[Tuple[int, float]]:
        """
        合并相近的匹配
        
        Args:
            matches: 匹配列表
            query_length: 查询序列长度
            
        Returns:
            List[Tuple[int, float]]: 合并后的匹配列表
        """
        if not matches:
            return []
            
        # 按位置排序
        position_sorted = sorted(matches, key=lambda x: x[0])
        position_groups = []
        current_group = [position_sorted[0]]
        
        # 合并相近的匹配
        min_group_distance = int(query_length * self.position_group_ratio)
        
        for match in position_sorted[1:]:
            if abs(match[0] - current_group[-1][0]) <= min_group_distance:
                current_group.append(match)
            else:
                position_groups.append(current_group)
                current_group = [match]
        
        position_groups.append(current_group)
        
        # 从每组中选择最佳匹配
        return [min(group, key=lambda x: x[1]) for group in position_groups]

    def visualize_matches(self, query: np.ndarray, matches: List[Tuple[int, float]], 
                         max_matches: int = 5, save_path: str = "images/matches.png"):
        """
        可视化查询序列和匹配结果
        
        Args:
            query: 查询序列
            matches: 匹配结果列表
            max_matches: 最多显示的匹配数量
            save_path: 图像保存路径
        """
        n_dims = query.shape[1]
        n_matches = min(len(matches), max_matches)
        
        if n_matches == 0:
            print("没有找到匹配结果")
            return
        
        # 创建images文件夹（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 创建图形布局
        fig = plt.figure(figsize=(15, 3 * (n_matches + 1)))
        gs = GridSpec(n_matches + 1, n_dims)
        fig.suptitle('匹配结果可视化 - DTW计算结果', fontsize=14, fontproperties=chinese_font)
        
        # 绘制查询序列
        for dim in range(n_dims):
            ax = fig.add_subplot(gs[0, dim])
            ax.plot(query[:, dim], 'b-', label='查询序列')
            ax.set_title(f'维度 {dim+1} - 查询序列', fontproperties=chinese_font)
            ax.legend(prop=chinese_font)
            ax.grid(True)
        
        # 绘制匹配序列
        for i, (pos, dist) in enumerate(matches[:max_matches]):
            for dim in range(n_dims):
                ax = fig.add_subplot(gs[i+1, dim])
                matched_seq = self.context_sequence[pos:pos+len(query), dim]
                ax.plot(matched_seq, 'r-', label=f'匹配 {i+1}')
                ax.set_title(f'维度 {dim+1} - 匹配 {i+1}\n位置: {pos}, DTW距离: {dist:.4f}', 
                           fontproperties=chinese_font)
                ax.legend(prop=chinese_font)
                ax.grid(True)
        
        plt.tight_layout()
        # 保存图像到指定路径
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"匹配结果图像已保存到: {save_path}") 