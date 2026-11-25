# 完整 11 标记联合 QC 的区域接口

`regional_qc.regional_joint_qc` 把完整实测面板的 `P(class AND QC)` 接到固定图 O/E、直接 NA 概率、条件比值矩和直接 CV。有限实现核验及全部失败见[区域报告](../../review/regional_joint_qc_2026-10-23/report.md)。这是独立分子 binomial rarefaction 的条件操作模型；固定总数 rarefaction、真实噪声校准、候选身份准确性和患者收益不在此接口的主张内。

```python
from spatial_membership_audit.regional_qc import regional_joint_qc

result = regional_joint_qc(
    counts, genes, baseline_eligible, 0.25, adjacency,
    pairs=((0, 1), (0, 3)),     # B–CD8, B–myeloid
    min_total=50, min_detected=20,
    accuracy_factor=1.25, max_seconds_per_pair=6.0,
)
```

`counts` 必须是 N×G 完整实测 Gene Expression 非负整数矩阵；`genes` 唯一并包含全部11 marker，未测基因不补零。保留率支持 scalar 或长度G的向量。`baseline_eligible` 是固定 bool mask；`adjacency` 为同N节点、二值对称无自环图。QC失败仍是图中的 OTHER，不删除节点或改变边。完整候选规则、输入条件及QC日志字段见[FULL_RULE_QC.md](FULL_RULE_QC.md)。`max_seconds_per_pair` 是算法边界检查的软上限；需要硬限时必须用外层进程监督器，并包含细胞概率计算时间。

图上的 O 是无向 A–B 边数，E=2M nA nB/[N(N−1)]，R=O/E。可用事件是 E>0；O=0 但 E>0 仍是有限 R=0。类别顺序为 B、CD8、CD4、myeloid 候选。

| 输出 | 解释 |
|---|---|
| `cells` | 每细胞联合QC、直接失败概率/log、四类概率及支持状态；不合格baseline行的有效状态是OTHER |
| `pairs[].availability` | 直接区域 `p_finite_direct`、`p_na`、ln概率及状态；结构零的log为None |
| `mean_R_interval` | E[R \| E>0] 的数值区间，文稿中的条件期望m₂；不是 E[O]/E[E] |
| `second_moment_R_interval`、`variance_R_interval` | 条件二阶矩及方差区间，未解决时可为None |
| `direct_cv.cv`、`cv_interval`、`status` | 唯一直接CV主值/区间/状态；没有Delta回填 |
| `ratio_of_means`、`raw_moments` | 浮点类别模型的 E[O]/E[E] 与原始O/E分量矩，供诊断，不冒充条件R均值 |

为了复用一次细胞计算，可以先调用 `candidate_joint_probabilities(..., include_partition=True)`，再调用 `summarize_joint_region(joint, adjacency, pairs=..., accuracy_factor=...)`。默认细胞API不启用额外partition，以保持历史输出和计算范围。两种区域入口均显式校验联合记录、类别概率与log分区相容。

## 直接补事件如何进入区域

QC通过时的OTHER由七个互不相交的marker门状态块正向累加：plasma阳性；B/T/M均阴性；各谱系冲突；单T但CD4/CD8均不满足或同时满足。再与直接QC失败的log概率合并，形成完整五类log分布。对任一A/B对，neither由其他三类直接相加。区域用四个不交状态“尚无A/B、仅A、仅B、两者都有”进行log递推；NA是前三者之和，不能用1−P+恢复。

这同时保持全规则与QC共享分子的依赖。不能将固定QC类别概率乘以独立的P(QC)替代联合计算。存在QC失败也不必然导致区域NA，接口在类别及区域层级分别计算两者。

## 状态与精度范围

- `STRUCTURAL_ZERO`：在所声明规则/操作的支持中不可达；空图必然没有有限O/E。
- `POSITIVE_FINITE` / `POSITIVE_SUBNORMAL`：可表示的正概率。
- `NUMERICAL_UNDERFLOW`：float=0但log有限，仍为可达事件。已支持QC failure log的区域传播。
- `UNRESOLVED_COMPONENT_UNDERFLOW`：类别或OTHER∩QC的可达质量本身下溢；保留细胞状态，拒绝将零交给区域结构判断。当前没有这些分量的一般log恢复。
- `UNRESOLVED_UPSTREAM_PRECISION`：直接补事件为正，却被单位浮点类别质量遮蔽。区域NA/log仍可用，但通用接口不据此给raw-count CV=0；mean/variance/CV主输出留空。
- `NO_FINITE_RATIO`：没有有限R；`ZERO_NUMERATOR_CV_UNDEFINED`：有限R恒为0，均值/方差为0，但CV未定义；`EXACT_ZERO_CV`：有限R恒为正数。
- `RESOLVED`：直接CV达到调用方的乘性精度目标；其他未解决状态原样保留，不由Delta或MC补齐。

数值包围只针对**输入binary64类别概率的固定图条件模型**，不覆盖上游概率求值、原始测量或候选身份误差。log分区本身也不是任意极小事件的严格相对误差证书。当前guard对简单恒比值图可能保守地不输出CV；这是已记录的范围限制。

## 本次有限验证

21项针对性检查通过，含完整小分子枚举、scalar/vector、精确零/非零CV、mask、非法输入及尾部参考。两历史完整区域共8条件已执行，512次MC所有draw保留。肺两对结构性不可用；肾p=.25两对可用率约0.422%/1.664%，CV约1.464/3425。另写边对参考路径支持两个肾模型端点的数值一致性；MC不足以核验CV。真实QC阈值50/20下两个构造尾部约9.56e−50/2.47e−360均获120/180位整数参考支持。完整来源、界限和失败保留见报告；不是整包回归或新队列验证。
