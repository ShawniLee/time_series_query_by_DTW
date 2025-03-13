import numpy as np
from time_series_matcher import TimeSeriesMatcher
import matplotlib.pyplot as plt
import os

def test_simple_pattern():
    """测试简单的正弦波模式匹配"""
    print("\n测试1: 简单正弦波模式")
    # 创建一个简单的查询模式：一个正弦波周期
    t_query = np.linspace(0, 2*np.pi, 50)
    query = np.column_stack([
        np.sin(t_query),
        np.cos(t_query)
    ])
    
    # 创建上下文序列：3个完整的周期
    t_context = np.linspace(0, 6*np.pi, 150)
    context = np.column_stack([
        np.sin(t_context),
        np.cos(t_context)
    ])
    
    # 期望找到的位置（每个周期的开始）
    expected_positions = [0, 50, 100]
    
    # 使用较小的阈值进行匹配
    matcher = TimeSeriesMatcher(context, threshold=.3, radius=2)
    matches, time_stats = matcher.find_matches(query)
    
    print(f"期望找到的位置: {expected_positions}")
    print(f"实际找到的位置: {[pos for pos, _ in matches]}")
    
    # 创建images文件夹（如果不存在）
    os.makedirs("images", exist_ok=True)
    
    # 可视化结果
    matcher.visualize_matches(query, matches, save_path="images/test1_simple_pattern.png")

def test_noisy_pattern():
    """测试带噪声的模式匹配"""
    print("\n测试2: 带噪声的模式")
    # 创建查询模式
    t_query = np.linspace(0, 2*np.pi, 50)
    query = np.column_stack([
        np.sin(t_query),
        np.cos(t_query)
    ])
    
    # 创建上下文序列：3个周期 + 噪声
    t_context = np.linspace(0, 6*np.pi, 150)
    context = np.column_stack([
        np.sin(t_context) + np.random.normal(0, 0.1, 150),
        np.cos(t_context) + np.random.normal(0, 0.1, 150)
    ])
    
    # 期望找到的位置
    expected_positions = [0, 50, 100]
    
    # 使用较大的阈值来处理噪声
    matcher = TimeSeriesMatcher(context, threshold=.5, radius=2)
    matches, time_stats = matcher.find_matches(query)
    
    print(f"期望找到的位置: {expected_positions}")
    print(f"实际找到的位置: {[pos for pos, _ in matches]}")
    
    # 可视化结果
    matcher.visualize_matches(query, matches, save_path="images/test2_noisy_pattern.png")

def test_scaled_pattern():
    """测试不同尺度的模式匹配"""
    print("\n测试3: 不同尺度的模式")
    # 创建查询模式
    t_query = np.linspace(0, 2*np.pi, 50)
    query = np.column_stack([
        np.sin(t_query),
        np.cos(t_query)
    ])
    
    # 创建上下文序列：包含不同尺度的模式
    t_context = np.linspace(0, 6*np.pi, 150)
    context = np.zeros((150, 2))
    
    # 添加三个不同尺度的模式
    scales = [1.0, 0.8, 1.2]
    positions = [0, 50, 100]
    
    for pos, scale in zip(positions, scales):
        context[pos:pos+50, 0] = scale * np.sin(t_query)
        context[pos:pos+50, 1] = scale * np.cos(t_query)
    
    # 使用较大的阈值来处理尺度变化
    matcher = TimeSeriesMatcher(context, threshold=1, radius=3)
    matches, time_stats = matcher.find_matches(query)
    
    print(f"期望找到的位置: {positions}")
    print(f"实际找到的位置: {[pos for pos, _ in matches]}")
    
    # 可视化结果
    matcher.visualize_matches(query, matches, save_path="images/test3_scaled_pattern.png")

def test_shifted_pattern():
    """测试时间偏移的模式匹配"""
    print("\n测试4: 时间偏移的模式")
    # 创建查询模式
    t_query = np.linspace(0, 2*np.pi, 50)
    query = np.column_stack([
        np.sin(t_query),
        np.cos(t_query)
    ])
    
    # 创建上下文序列：包含时间偏移的模式
    t_context = np.linspace(0, 6*np.pi, 150)
    context = np.zeros((150, 2))
    
    # 添加三个不同时间偏移的模式
    shifts = [0, np.pi/4, -np.pi/4]
    positions = [0, 50, 100]
    
    for pos, shift in zip(positions, shifts):
        context[pos:pos+50, 0] = np.sin(t_query + shift)
        context[pos:pos+50, 1] = np.cos(t_query + shift)
    
    # 使用较大的radius来处理时间偏移
    matcher = TimeSeriesMatcher(context, threshold=.5, radius=5)
    matches, time_stats = matcher.find_matches(query)
    
    print(f"期望找到的位置: {positions}")
    print(f"实际找到的位置: {[pos for pos, _ in matches]}")
    
    # 可视化结果
    matcher.visualize_matches(query, matches, save_path="images/test4_shifted_pattern.png")

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 运行所有测试
    test_simple_pattern()
    test_noisy_pattern()
    test_scaled_pattern()
    test_shifted_pattern() 