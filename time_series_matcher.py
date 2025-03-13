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
from time_series_matcher_class import TimeSeriesMatcher, configure_chinese_font, chinese_font

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def visualize_example_data(query: np.ndarray, context: np.ndarray, pattern_positions: List[int]):
    """
    可视化示例数据，包括查询序列和带有标记匹配位置的上下文序列
    
    Args:
        query: 查询序列
        context: 上下文序列
        pattern_positions: 模式的实际位置列表
    """
    n_dims = query.shape[1]
    
    # 创建两行图表：上面是查询序列，下面是完整上下文序列和模式位置
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2 + len(pattern_positions), 1)
    fig.suptitle('示例数据可视化', fontsize=14, fontproperties=chinese_font)
    
    # 绘制查询序列
    ax = fig.add_subplot(gs[0])
    for dim in range(n_dims):
        ax.plot(query[:, dim], label=f'维度 {dim+1}')
    ax.set_title('查询序列（衰减的正弦/余弦波）', fontproperties=chinese_font)
    ax.legend(prop=chinese_font)
    ax.grid(True)
    
    # 绘制上下文序列的一部分（为了可视化效果，只显示前2000个点）
    ax = fig.add_subplot(gs[1])
    display_length = 10000  # 显示更多的上下文序列
    for dim in range(n_dims):
        ax.plot(context[:display_length, dim], label=f'维度 {dim+1}', alpha=0.7)
    
    # 标记模式位置
    for pos in pattern_positions:
        if pos < display_length:
            ax.axvspan(pos, pos + len(query), color='yellow', alpha=0.3)
    
    ax.set_title('上下文序列（显示前10000个点）- 黄色区域标记了模式位置', 
                 fontproperties=chinese_font)
    ax.legend(prop=chinese_font)
    ax.grid(True)
    
    # 添加每个模式位置的详细视图
    for i, pos in enumerate(pattern_positions):
        ax = fig.add_subplot(gs[i+2])
        for dim in range(n_dims):
            # 绘制模式
            pattern_seq = context[pos:pos+len(query), dim]
            ax.plot(pattern_seq, 'r-', label=f'模式 {i+1} 维度 {dim+1}')
            # 绘制查询序列作为比较
            ax.plot(query[:, dim], 'b--', label=f'查询序列 维度 {dim+1}', alpha=0.7)
        
        ax.set_title(f'模式 {i+1} 位置: {pos} - 与查询序列形状比较', fontproperties=chinese_font)
        ax.legend(prop=chinese_font)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_time_stats(time_stats: Dict[str, float]):
    """
    可视化时间统计数据
    
    Args:
        time_stats: 包含各阶段耗时的字典
    """
    # 提取各阶段时间和总时间
    total_time = time_stats["total"]
    stage_times = {k: v for k, v in time_stats.items() if k != "total"}
    
    # 计算各阶段占总时间的百分比
    percentages = {k: (v / total_time) * 100 for k, v in stage_times.items()}
    
    # 创建饼图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 绘制各阶段耗时柱状图
    bars = ax1.bar(stage_times.keys(), stage_times.values())
    ax1.set_title('各阶段耗时(秒)', fontproperties=chinese_font)
    ax1.set_ylabel('时间 (秒)', fontproperties=chinese_font)
    ax1.tick_params(axis='x', rotation=45)
    
    # 在柱子上方显示具体数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}s',
                ha='center', va='bottom', rotation=0, fontproperties=chinese_font)
    
    # 绘制各阶段占比饼图
    wedges, texts, autotexts = ax2.pie(
        percentages.values(), 
        labels=percentages.keys(),
        autopct='%1.1f%%',
        startangle=90
    )
    # 设置饼图中文字体
    for text in texts:
        text.set_fontproperties(chinese_font)
    for autotext in autotexts:
        autotext.set_fontproperties(chinese_font)
        
    ax2.set_title('各阶段占比', fontproperties=chinese_font)
    ax2.axis('equal')  # 确保饼图是圆的而不是椭圆的
    
    # 添加总时间文本
    plt.figtext(0.5, 0.01, f'总耗时: {total_time:.4f}秒', ha='center', 
               fontproperties=chinese_font, fontsize=12)
    
    plt.tight_layout()
    plt.show()

def demo():
    """演示使用示例"""
    # 生成示例数据
    np.random.seed(42)
    
    # 创建查询序列：一个包含特定模式的二维时间序列
    t = np.linspace(0, 4*np.pi, 100)
    pattern = np.exp(-t/10)  # 衰减因子
    query = np.column_stack([
        pattern * np.sin(t),  # 衰减的正弦波
        pattern * np.cos(t)   # 衰减的余弦波
    ])
    
    # 创建上下文序列：在长序列中嵌入多个相似模式
    t_long = np.linspace(0, 40*np.pi, 40000)
    base_signal = np.column_stack([
        np.sin(t_long),
        np.cos(t_long)
    ])
    
    # 添加随机噪声
    noise = np.random.normal(0, 0.1, base_signal.shape)
    
    # 在特定位置嵌入查询模式（添加一些变化）
    context = base_signal + noise
    pattern_positions = [1000, 5000, 15000, 25000, 35000]
    
    for pos in pattern_positions:
        if pos + len(query) <= len(context):
            # 在每个位置添加稍微变形的模式
            variation = np.random.uniform(0.8, 1.2)  # 随机振幅变化
            time_shift = np.random.randint(-5, 5)    # 随机时间偏移
            pattern_segment = np.column_stack([
                variation * pattern * np.sin(t + time_shift/10),
                variation * pattern * np.cos(t + time_shift/10)
            ])
            context[pos:pos+len(query)] = pattern_segment + noise[pos:pos+len(query)] * 0.5
    
    # 可视化示例数据
    print("显示示例数据...")
    visualize_example_data(query, context, pattern_positions)
    
    # 初始化匹配器
    matcher = TimeSeriesMatcher(
        context, 
        threshold=0.5,        # 降低DTW距离阈值
        radius=1,             # DTW warping半径
        position_group_ratio=0.1,  # 位置分组比例
        lb_keogh_multiplier=1.5    # LB_Keogh剪枝倍数
    )

    
    
    print("\n开始查找匹配...")
    # 查找匹配
    matches, time_stats = matcher.find_matches(query)
    
    # 显示时间统计
    print("\n时间统计:")
    for stage, time_spent in time_stats.items():
        print(f"  {stage}: {time_spent:.4f}秒")
    
    # 可视化时间统计
    visualize_time_stats(time_stats)
    
    print(f"\n找到 {len(matches)} 个匹配")
    if len(matches) > 0:
        print("\n前5个最佳匹配：")
        for pos, dist in matches[:5]:
            print(f"位置: {pos}, DTW距离: {dist:.4f}")
        
        # 计算匹配位置与实际嵌入位置的对比
        found_positions = set(pos for pos, _ in matches)
        actual_matches = sum(1 for pos in pattern_positions 
                           if any(abs(pos - found_pos) < 100 for found_pos in found_positions))
        print(f"\n在{len(pattern_positions)}个实际模式中成功找到了{actual_matches}个")
    
    # 可视化匹配结果
    print("\n显示匹配结果...")
    matcher.visualize_matches(query, matches)

if __name__ == "__main__":
    demo() 