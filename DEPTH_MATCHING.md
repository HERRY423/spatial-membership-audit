# 逐基因双向匹配与过离散敏感性

`depth_matching.bidirectional_retention` 对齐两组每基因均值，返回两个 `[0,1]` 保留率向量。每个基因只稀释均值较高的一侧，使两侧的条件期望等于较小均值。它匹配**观测均值**，不估计真实检测效率；调用者须先对齐基因身份、计数单位和校准范围。

```python
from spatial_membership_audit.depth_matching import (
    bidirectional_retention, thin_histogram_summary,
)

fit = bidirectional_retention([8, 2, 4, 0], [2, 8, 4, 0])
assert list(fit.retention_a) == [0.25, 1.0, 1.0, 1.0]
assert list(fit.retention_b) == [1.0, 0.25, 1.0, 1.0]

# Each histogram bin is the number of cells with count 0, 1, 2, ...
binomial = thin_histogram_summary([7, 5, 0, 2, 3], 0.25)
overdispersed = thin_histogram_summary([7, 5, 0, 2, 3], 0.25, rho=0.10)
```

将两个保留率向量分别传给已有接受逐基因 `p` 的 analytic/marker 接口，即可按两侧自身计数执行 binomial 计算。此模块不修改旧的 `shrink_depth_retention` 或现有分析默认参数。若方向和比率由同一对样本的均值拟合，均值相等是恒等式，不能用作验证；若从校准样本估计，应冻结后原样用于评估，并报告方向反转，不能悄悄重新拟合。

边界状态：相等正均值为 `TIED`，两侧均保留；双零为 `BOTH_ZERO`，两侧p=1且效率无从识别；单零为 `COMMON_TARGET_ZERO`，仅有计数侧p=0。没有伪计数、插补或删除不合适的基因。缺失、负值或非有限值报错。

`thin_histogram_summary` 返回 mean、variance、zero_fraction。variance 是经验细胞分布的总体方差（分母N），不是均值估计误差或空间O/E的方差。rho=0为Binomial；rho>0为cell×gene保留概率的Beta混合，条件方差为 `X*p*(1-p)*(1+(X-1)*rho)`。p=0/1有明确退化处理。rho需由调用者指定；本接口不拟合技术噪声，且没有共享细胞/批次因子的相关结构。

**过离散接口目前只覆盖计数直方图统计量。** 不得把其结果直接称为原binomial图算法的P+、O/E、CV或假阳性率验证；这些空间端点需要相应概率模型及单独测试。

真实14切片/7患者的回顾性诊断及完整固定rho网格：`review/bidirectional_matching_2026-11-01/REPORT.md`（工作区根目录下）。工程验证：`tests/test_depth_matching.py`。
