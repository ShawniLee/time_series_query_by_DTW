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

def visualize_example_data(query: np.ndarray, context: np.ndarray, pattern_positions: List[int], 
                       save_path: str = "images/example_data.png"):
    """
    可视化示例数据，包括查询序列和带有标记匹配位置的上下文序列
    
    Args:
        query: 查询序列
        context: 上下文序列
        pattern_positions: 模式的实际位置列表
        save_path: 图像保存路径
    """
    n_dims = query.shape[1]
    
    # 创建images文件夹（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
    # 保存图像到指定路径
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"示例数据图像已保存到: {save_path}")

def visualize_time_stats(time_stats: Dict[str, float], save_path: str = "images/time_stats.png"):
    """
    可视化时间统计数据
    
    Args:
        time_stats: 包含各阶段耗时的字典
        save_path: 图像保存路径
    """
    # 创建images文件夹（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
    # 保存图像到指定路径
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"时间统计图像已保存到: {save_path}")

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
    visualize_example_data(query, context, pattern_positions, "images/demo_example_data.png")
    
    # 初始化匹配器
    matcher = TimeSeriesMatcher(
        context, 
        threshold=0.5,
        radius=1,
        position_group_ratio=0.1,
        lb_keogh_multiplier=1.2,    # 降低multiplier以提高剪枝效率
        downsample_factor=2         # 使用2倍降采样
    )

    
    
    print("\n开始查找匹配...")
    # 查找匹配
    matches, time_stats = matcher.find_matches(query)
    
    # 显示时间统计
    print("\n时间统计:")
    for stage, time_spent in time_stats.items():
        print(f"  {stage}: {time_spent:.4f}秒")
    
    # 可视化时间统计
    visualize_time_stats(time_stats, "images/demo_time_stats.png")
    
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
    matcher.visualize_matches(query, matches, save_path="images/demo_matches.png")

def visualize_multi_query_matches(context: np.ndarray, queries: List[np.ndarray], 
                               all_matches: List[List[Tuple[int, float]]], 
                               labels: List[str] = None,
                               save_path: str = "images/multi_query_matches.png"):
    """
    可视化多查询匹配结果
    
    Args:
        context: 上下文序列
        queries: 查询序列列表
        all_matches: 每个查询序列对应的匹配结果列表
        labels: 查询序列的标签列表
        save_path: 图像保存路径
    """
    n_queries = len(queries)
    if n_queries == 0:
        print("没有查询序列")
        return
    
    if labels is None:
        labels = [f"查询 {i+1}" for i in range(n_queries)]
    
    # 创建images文件夹（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 准备彩色映射表，为每个查询分配不同颜色
    cmap = plt.cm.get_cmap('tab10', n_queries)
    colors = [cmap(i) for i in range(n_queries)]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('多查询匹配结果', fontsize=14, fontproperties=chinese_font)
    
    # 绘制所有查询序列
    for i, query in enumerate(queries):
        n_dims = query.shape[1]
        for dim in range(n_dims):
            ax1.plot(query[:, dim], color=colors[i], 
                    label=f'{labels[i]} - 维度 {dim+1}', 
                    linestyle=['-', '--', '-.', ':'][dim % 4])
    
    ax1.set_title('所有查询序列', fontproperties=chinese_font)
    ax1.legend(prop=chinese_font)
    ax1.grid(True)
    
    # 绘制上下文序列和匹配位置
    display_length = min(10000, len(context))  # 显示更多的上下文序列
    for dim in range(context.shape[1]):
        ax2.plot(context[:display_length, dim], 'k-', 
                alpha=0.5, label=f'上下文序列 - 维度 {dim+1}')
    
    # 为每个查询添加不同颜色的匹配区域
    for i, matches in enumerate(all_matches):
        if not matches:
            continue
            
        query_length = len(queries[i])
        # 只显示前20个匹配，避免图表太拥挤
        for pos, dist in matches[:20]:
            if pos < display_length:
                ax2.axvspan(pos, pos + query_length, color=colors[i], 
                           alpha=0.3, label=f'{labels[i]} 匹配' if pos == matches[0][0] else "")
                # 添加位置和距离文本
                ax2.text(pos, 0.95 - 0.05*i, f"{pos}\n{dist:.2f}", 
                        fontsize=8, verticalalignment='top', 
                        horizontalalignment='left', 
                        color=colors[i], backgroundcolor='w',
                        transform=ax2.get_xaxis_transform())
    
    ax2.set_title('上下文序列与匹配位置 (显示前10000个点)', fontproperties=chinese_font)
    ax2.legend(prop=chinese_font, loc='upper right')
    ax2.grid(True)
    
    # 添加结果摘要文本
    summary_text = "匹配结果摘要:\n"
    for i, matches in enumerate(all_matches):
        summary_text += f"{labels[i]}: 找到 {len(matches)} 个匹配\n"
    
    plt.figtext(0.5, 0.01, summary_text, ha='center', 
               fontproperties=chinese_font, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 为底部文本留出空间
    # 保存图像到指定路径
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"多查询匹配结果图像已保存到: {save_path}")

def multi_query_demo():
    """多查询功能演示"""
    # 生成示例数据
    np.random.seed(42)
    
    # 创建多个查询序列
    queries = []
    labels = []
    
    # 查询1：衰减的正弦/余弦波（与原demo相同）
    t1 = np.linspace(0, 4*np.pi, 100)
    pattern1 = np.exp(-t1/10)  # 衰减因子
    query1 = np.column_stack([
        pattern1 * np.sin(t1),  # 衰减的正弦波
        pattern1 * np.cos(t1)   # 衰减的余弦波
    ])
    queries.append(query1)
    labels.append("衰减波")
    
    # 查询2：三角波
    t2 = np.linspace(0, 4*np.pi, 120)
    query2 = np.column_stack([
        np.abs((t2 / np.pi) % 2 - 1),  # 三角波
        np.sin(t2) * 0.5                # 较小振幅的正弦波
    ])
    queries.append(query2)
    labels.append("三角波")
    
    # 查询3：方波
    t3 = np.linspace(0, 4*np.pi, 80)
    query3 = np.column_stack([
        np.sign(np.sin(t3)),           # 方波
        np.cos(2*t3) * 0.5             # 高频余弦波
    ])
    queries.append(query3)
    labels.append("方波")
    
    # 创建上下文序列：在长序列中嵌入多个相似模式
    t_long = np.linspace(0, 40*np.pi, 40000)
    base_signal = np.column_stack([
        np.sin(t_long),
        np.cos(t_long)
    ])
    
    # 添加随机噪声
    noise = np.random.normal(0, 0.1, base_signal.shape)
    context = base_signal + noise
    
    # 嵌入不同的模式
    pattern_positions = {
        "衰减波": [1000, 15000, 35000],
        "三角波": [5000, 20000, 30000],
        "方波": [10000, 25000, 38000]
    }
    
    # 嵌入查询1（衰减波）
    t1 = np.linspace(0, 4*np.pi, 100)
    pattern1 = np.exp(-t1/10)
    for pos in pattern_positions["衰减波"]:
        if pos + len(query1) <= len(context):
            variation = np.random.uniform(0.8, 1.2)
            time_shift = np.random.randint(-5, 5)
            pattern_segment = np.column_stack([
                variation * pattern1 * np.sin(t1 + time_shift/10),
                variation * pattern1 * np.cos(t1 + time_shift/10)
            ])
            context[pos:pos+len(query1)] = pattern_segment + noise[pos:pos+len(query1)] * 0.5
    
    # 嵌入查询2（三角波）
    for pos in pattern_positions["三角波"]:
        if pos + len(query2) <= len(context):
            variation = np.random.uniform(0.9, 1.1)
            pattern_segment = np.column_stack([
                variation * np.abs((t2 / np.pi) % 2 - 1),
                variation * np.sin(t2) * 0.5
            ])
            context[pos:pos+len(query2)] = pattern_segment + noise[pos:pos+len(query2)] * 0.5
    
    # 嵌入查询3（方波）
    for pos in pattern_positions["方波"]:
        if pos + len(query3) <= len(context):
            variation = np.random.uniform(0.85, 1.15)
            pattern_segment = np.column_stack([
                variation * np.sign(np.sin(t3)),
                variation * np.cos(2*t3) * 0.5
            ])
            context[pos:pos+len(query3)] = pattern_segment + noise[pos:pos+len(query3)] * 0.5
    
    # 可视化多查询示例数据
    print("生成并显示多查询示例数据...")
    
    # 为每个查询类型创建可视化
    for i, (query_name, positions) in enumerate(pattern_positions.items()):
        query = queries[i]
        save_path = f"images/multi_query_demo_{query_name}_data.png"
        visualize_example_data(query, context, positions, save_path)
    
    # 初始化匹配器
    matcher = TimeSeriesMatcher(
        context, 
        threshold=0.6,  # 稍微放宽阈值，以适应不同类型的模式
        radius=2,      # 增加半径以提高灵活性
        position_group_ratio=0.1,
        lb_keogh_multiplier=1.2,
        downsample_factor=2
    )
    
    # 存储所有查询的匹配结果
    all_matches = []
    total_time_stats = {"total": 0}
    
    # 对每个查询序列执行匹配
    print("\n开始执行多查询匹配...")
    for i, query in enumerate(queries):
        print(f"\n处理查询 {i+1}: {labels[i]}")
        
        # 查找匹配
        matches, time_stats = matcher.find_matches(query)
        all_matches.append(matches)
        
        # 累加总时间统计
        for key, value in time_stats.items():
            if key in total_time_stats:
                total_time_stats[key] += value
            else:
                total_time_stats[key] = value
        
        # 显示每个查询的匹配结果
        print(f"  找到 {len(matches)} 个匹配")
        if len(matches) > 0:
            print(f"  前3个最佳匹配：")
            for pos, dist in matches[:3]:
                print(f"  位置: {pos}, DTW距离: {dist:.4f}")
        
        # 可视化这个查询的匹配结果
        matcher.visualize_matches(query, matches, 
                                save_path=f"images/multi_query_demo_{labels[i]}_matches.png")
    
    # 显示总时间统计
    print("\n总时间统计:")
    for stage, time_spent in total_time_stats.items():
        print(f"  {stage}: {time_spent:.4f}秒")
    
    # 可视化总时间统计
    visualize_time_stats(total_time_stats, "images/multi_query_demo_time_stats.png")
    
    # 计算每个查询的匹配准确率
    for i, query_name in enumerate(pattern_positions.keys()):
        true_positions = pattern_positions[query_name]
        matches = all_matches[i]
        if not matches:
            print(f"\n{query_name}: 未找到匹配")
            continue
            
        found_positions = set(pos for pos, _ in matches)
        actual_matches = sum(1 for pos in true_positions 
                          if any(abs(pos - found_pos) < 100 for found_pos in found_positions))
        print(f"\n{query_name}: 在{len(true_positions)}个实际模式中成功找到了{actual_matches}个")
    
    # 可视化所有查询的综合匹配结果
    visualize_multi_query_matches(context, queries, all_matches, labels=labels, 
                              save_path="images/multi_query_demo_all_matches.png")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        multi_query_demo()
    else:
        demo() 