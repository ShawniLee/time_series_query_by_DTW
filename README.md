# 时间序列查询匹配器

这个项目实现了一个高效的时间序列子序列匹配系统，支持在长序列中查找与给定查询序列相似的子序列。该实现特别适用于多维时间序列数据，并且支持非严格的时间对齐。

## 主要特性

- 支持多维时间序列数据
- Z-score标准化预处理
- 使用LB_Keogh下界进行快速剪枝
- 基于DTW（动态时间规整）的相似度计算
- 支持非严格时间对齐的匹配
- 支持多个匹配结果

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用示例

```python
import numpy as np
from time_series_matcher import TimeSeriesMatcher

# 创建示例数据
context = np.random.randn(40000, 2)  # 上下文序列
query = np.random.randn(100, 2)      # 查询序列

# 初始化匹配器
matcher = TimeSeriesMatcher(context, threshold=1.0)

# 查找匹配
matches = matcher.find_matches(query)

# 打印结果
print(f"找到 {len(matches)} 个匹配")
for pos, dist in matches[:5]:
    print(f"位置: {pos}, DTW距离: {dist:.4f}")
```

## 实现细节

1. **预处理**：对查询序列和上下文序列的每个维度进行Z-score标准化，确保比较的公平性。

2. **高效搜索**：
   - 使用LB_Keogh下界快速剪枝，减少不必要的DTW计算
   - 支持设置DTW warping半径，平衡精度和效率

3. **相似度计算**：
   - 使用FastDTW算法进行高效的DTW距离计算
   - 支持多维时间序列的比较

4. **结果收集**：
   - 返回所有DTW距离小于阈值的匹配
   - 按相似度排序输出结果 