# 重新检查 QC 的解析 rarefaction 与置换 z 接口

这是可选的开发接口。原有 fixed-QC 接口和已冻结结果保留。四 marker 实现的有限验证与完整失败记录见 [R1–R4 报告](../../review/rarefaction_r1_r4_2026-10-23/report.md)。2026-10-23 另已新增[完整 11 marker 联合 QC 接口](FULL_RULE_QC.md)，含直接 QC 失败补事件与首个真实故障的精确参考；下文四 marker 接口的原有范围不变。

## 联合 QC 操作

`joint_qc.four_gene_joint_qc` 对单个细胞计算独立分子保留后，类别条件与总 counts/检出基因门槛同时满足的概率。输入必须包含完整实测 Gene Expression 面板；调用方负责确认基因名称、面板完整性及四个 marker 下标，未测基因不得补零。

```python
from spatial_membership_audit.joint_qc import four_gene_joint_qc

result = four_gene_joint_qc(
    [1, 1, 0, 0], 0.5,
    marker_indices=(0, 1, 2, 3), min_total=2, min_detected=2,
)
# p_a_and_qc = 0.25; p_qc = 0.25; p_a_without_qc = 0.25
# p_qc * p_a_without_qc = 0.0625 is not the joint probability.
```

A 要求前两个 marker 都检出，B 要求后两个都检出；A/B 同时满足时互相 veto，归 OTHER。每基因保留率可为一个 scalar 或对齐的向量。共同 p 下总 counts 为 Binomial；不同 p_g 下总数为独立 Binomial 之和。检出基因数是 Poisson-binomial，但它与总 counts、marker 分类共享计数，不能将各自边际概率相乘。

实现对总 counts 与检出基因数作门槛截断 DP，保留四 marker 的 16 种检出状态。函数返回 `p_a_and_qc`、`p_b_and_qc`、`p_other_and_qc`、`p_qc`、`a_status`、`b_status`、`qc_status` 和质量残差。该函数仍为四 marker 模型；完整 11 marker classifier 另由 `candidate_qc` 接口实现。

把输出接到固定图概率引擎时，A/B 使用联合概率；OTHER 概率含未过 QC 的全部质量。**QC 失败仍是固定图的 OTHER 节点，N、M 均不变。** 若研究操作要求删除失败细胞、重建图或更换分母，不能直接使用本接口冒充该操作。

`STRUCTURAL_ZERO` 根据原始 counts/p 的可达支持判定；可达但浮点值为零返回 `NUMERICAL_UNDERFLOW`；正值另分 `POSITIVE_SUBNORMAL` 与 `POSITIVE_FINITE`。DP 为 binary64，质量残差上限 2e-12。它不提供极小联合尾部的相对误差证书或 log 输出；下游必须保留这些状态，不能将 underflow 当结构零。已有 availability 的 log-tail 路线与本 DP 的上游概率精度范围应分开。

## 固定类别数的 O/E → z

`permutation_bridge.fixed_size_edge_moments` 用精确有理数计算无向简单二值图中，固定 n_A/n_B 均匀随机置换节点标签的 E[O] 和 Var(O)。`observed_permutation_bridge` 同时提供 hard-label 观察 O、R=O/E 和 z。

令 N 为节点数、M 为边数、S 为共享一个端点的无序边对数、D 为互不相交边对数，(x)_k 为下降阶乘：

```
q1 = 2*n_A*n_B/(N)_2
q2 = n_A*n_B*(n_A+n_B-2)/(N)_3
q3 = 4*(n_A)_2*(n_B)_2/(N)_4
E = M*q1
Var(O) = E + 2*S*q2 + 2*D*q3 - E**2
z = (O-E)/sqrt(Var(O)) = (R-1)*E/sqrt(Var(O))
```

不足相应节点数的事件项为零。E=0 时 R 未定义；Var=0 时 z 未定义，例如 complete graph 即使 R=1，也不能强行填 z=0。z 依赖图结构与类别数，单凭 R 没有通用转换。随机 rarefaction 后类别数也变化，不能直接把其均值 R 代入这个固定类别数零分布当作 E[z]。

该 null 对应本轮本地 Squidpy 1.6.5 节点置换；Giotto 4.2.3 被调用实现使用端点标签池置换，单独报告，不能用上述公式作其真值。原生方法的版本、参数、完整输出和 installed source 均保存。空间 bootstrap 的组织支持域不由本接口提供。

## 验证范围

200 项有限工程检查通过；QC 完整计数 Fraction 枚举最大绝对误差 6.66e-16，置换矩在 N≤6 图上完整标签枚举一致。两个历史肺/肾窗口 p=.25 的联合可用率与各自 512 次原始分子抽样相容。完整 97 项回归首次有 7 项临时目录权限错误，移入工作区临时目录后仅重测原 7 项并全部通过；全部日志保留。这些是所定义操作的数值证据，不是物理重复校准、身份准确性或队列效用证据。
