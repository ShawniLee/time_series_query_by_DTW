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
                 lb_keogh_multiplier: float = 1.5):
        """
        初始化时间序列匹配器
        
        Args:
            context_sequence: 形状为 (length, dimensions) 的上下文序列
            threshold: DTW距离阈值，用于确定匹配
            radius: DTW warping半径，影响DTW计算的弹性程度
            position_group_ratio: 用于确定相邻匹配分组的比例，值越小分组越严格
            lb_keogh_multiplier: LB_Keogh剪枝倍数，值越小剪枝越严格

            radius 经验值：
            对于大多数应用，radius = 1-10是合理的
            较短序列（100个点以下）可以用1-5
            较长序列可以用较大的值（5-20）
            代码中默认值为1，这是一个保守设置，适合于对时间扭曲要求不高但计算效率要求高的场景

            LB_Keogh剪枝原理分析
            LB_Keogh算法是一种用于加速DTW计算的下界估计方法：
            它计算查询序列与上下文窗口的上下包络线之间的距离
            由于LB_Keogh ≤ 真实DTW距离，可以用它来快速过滤掉不可能匹配的序列
            lb_keogh_multiplier调整了这个剪枝的松紧度：
            较大的值意味着更宽松的剪枝（更多候选序列进入DTW计算阶段）
            较小的值意味着更严格的剪枝（更少候选序列进入DTW计算阶段）
            合理范围
            理论下限：1.0（LB_Keogh严格等于阈值时剪枝）
            实际范围：通常在1.0-5.0之间
            权衡考虑：
            设置过小（接近1.0）：剪枝非常严格，可能会错过一些真实匹配
            设置过大（超过5.0）：剪枝过于宽松，可能会增加不必要的计算量
            默认值1.5：平衡计算效率和匹配准确性   
        """
        self.context_sequence = context_sequence
        self.threshold = threshold
        self.radius = radius
        self.position_group_ratio = position_group_ratio
        self.lb_keogh_multiplier = lb_keogh_multiplier
    
    def _compute_lb_keogh(self, query: np.ndarray, window: np.ndarray) -> float:
        """
        计算LB_Keogh下界
        
        Args:
            query: 查询序列
            window: 待比较的窗口
            
        Returns:
            float: LB_Keogh下界值
        """
        length = len(query)
        n_dims = query.shape[1]
        upper_env = np.zeros_like(query)
        lower_env = np.zeros_like(query)
        
        # 计算每个维度的上下包络
        for i in range(length):
            start = max(0, i - self.radius)
            end = min(length, i + self.radius + 1)
            upper_env[i] = np.max(window[start:end], axis=0)
            lower_env[i] = np.min(window[start:end], axis=0)
        
        # 计算每个维度的距离并归一化
        dim_dists = np.zeros(n_dims)
        for dim in range(n_dims):
            query_dim = query[:, dim]
            upper_env_dim = upper_env[:, dim]
            lower_env_dim = lower_env[:, dim]
            
            # 计算超出包络的距离
            dim_dists[dim] = np.sum(np.where(query_dim > upper_env_dim,
                                     (query_dim - upper_env_dim) ** 2,
                                     np.where(query_dim < lower_env_dim,
                                            (query_dim - lower_env_dim) ** 2,
                                            0)))
        
        # 使用欧几里得范数计算总距离
        dist = np.sqrt(np.sum(dim_dists) / length)
        
        return dist
    
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
        # 初始化时间统计
        time_stats = {
            "lb_keogh_pruning": 0,
            "dtw_calculation": 0,
            "post_processing": 0,
            "total": 0
        }
        
        total_start_time = time.time()
        
        query_length = len(query)
        matches = []
        
        # 滑动窗口搜索
        logger.info(f"开始查找匹配，查询序列长度: {query_length}，阈值: {self.threshold}")
        
        lb_keogh_total = 0
        dtw_total = 0
        
        for i in range(len(self.context_sequence) - query_length + 1):
            window = self.context_sequence[i:i+query_length]
            
            # 使用LB_Keogh进行快速剪枝
            lb_start = time.time()
            lb_dist = self._compute_lb_keogh(query, window)
            lb_time = time.time() - lb_start
            lb_keogh_total += lb_time
            
            # 记录LB_Keogh剪枝结果
            if lb_dist > self.threshold * self.lb_keogh_multiplier:
                logger.debug(f"位置 {i}: LB_Keogh距离 {lb_dist:.4f} > {self.threshold * self.lb_keogh_multiplier:.4f}，剪枝跳过")
                continue
            else:
                logger.debug(f"位置 {i}: LB_Keogh距离 {lb_dist:.4f} <= {self.threshold * self.lb_keogh_multiplier:.4f}，计算DTW")
            
            # 对通过剪枝的窗口计算完整DTW距离
            dtw_start = time.time()
            distance, path = fastdtw(query, window, radius=self.radius)
            dtw_time = time.time() - dtw_start
            dtw_total += dtw_time
            
            # 计算归一化DTW距离 - 使用路径长度而非固定值进行归一化
            path_length = len(path)
            distance = np.sqrt(distance / path_length)
            
            # 记录DTW距离结果
            if distance <= self.threshold:
                matches.append((i, distance))
                logger.info(f"找到匹配: 位置 {i}, DTW距离: {distance:.4f}")
            else:
                logger.debug(f"位置 {i}: DTW距离 {distance:.4f} > {self.threshold:.4f}，不符合阈值要求")
        
        time_stats["lb_keogh_pruning"] = lb_keogh_total
        time_stats["dtw_calculation"] = dtw_total
        
        # 记录所有原始匹配
        logger.info(f"合并前的原始匹配: {[(pos, f'{dist:.4f}') for pos, dist in matches]}")
        
        # 后处理：合并相近的匹配
        post_start = time.time()
        merged_matches = []
        if matches:
            # 按位置分组 - 先将位置接近的匹配放在一起
            position_groups = []
            # 首先按位置排序
            position_sorted = sorted(matches, key=lambda x: x[0])
            
            if position_sorted:
                current_group = [position_sorted[0]]
                
                for match in position_sorted[1:]:
                    # 使用位置分组比例参数计算最小距离
                    min_group_distance = int(query_length * self.position_group_ratio)
                    
                    # 如果当前匹配与组内最后一个匹配的位置足够近，加入同一组
                    if abs(match[0] - current_group[-1][0]) <= min_group_distance:
                        current_group.append(match)
                        logger.info(f"分组: 将位置 {match[0]} (距离 {match[1]:.4f}) 加入到与位置 {current_group[0][0]} 相同的组")
                    else:
                        # 否则创建新组
                        position_groups.append(current_group)
                        logger.info(f"分组: 完成一个组，包含 {len(current_group)} 个匹配，起始位置 {current_group[0][0]}")
                        current_group = [match]
                
                # 添加最后一组
                position_groups.append(current_group)
                logger.info(f"分组: 完成最后一个组，包含 {len(current_group)} 个匹配，起始位置 {current_group[0][0]}")
            
            # 从每组中选择距离最小的匹配
            for group in position_groups:
                best_match = min(group, key=lambda x: x[1])
                merged_matches.append(best_match)
                logger.info(f"合并: 从包含位置 {[pos for pos, _ in group]} 的组中选择位置 {best_match[0]} (距离 {best_match[1]:.4f})")
        
        time_stats["post_processing"] = time.time() - post_start
        time_stats["total"] = time.time() - total_start_time
        
        logger.info(f"合并后的最终匹配: {[(pos, f'{dist:.4f}') for pos, dist in merged_matches]}")
        return merged_matches, time_stats

    def visualize_matches(self, query: np.ndarray, matches: List[Tuple[int, float]], 
                         max_matches: int = 5):
        """
        可视化查询序列和匹配结果
        
        Args:
            query: 查询序列
            matches: 匹配结果列表
            max_matches: 最多显示的匹配数量
        """
        n_dims = query.shape[1]
        n_matches = min(len(matches), max_matches)
        
        if n_matches == 0:
            print("没有找到匹配结果")
            return
        
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
        plt.show() 