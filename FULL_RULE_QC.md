# 完整 11 marker 分类规则的联合 QC

同日最新：[区域接口](REGIONAL_QC.md)已将该联合概率和直接补事件log接到固定图O/E、可用率及直接CV；两历史完整区域、8条件与512次分子MC的有限验证见[报告](../../review/regional_joint_qc_2026-10-23/report.md)。下文“实际验证范围”记录之前的细胞级核验；区域结果与其分别保留。

2026-10-23 开发接口；有限验证和失败记录见[本轮报告](../../review/joint_qc_full_rule_2026-10-23/report.md)。`candidate_qc` 将原 `common125_candidate_v1` 与下采样后的 counts/检出基因 QC 作为一个联合事件计算。四 marker 接口和此前固定 QC 操作各自保留。

同日后续[数值修复报告](../../review/joint_qc_tail_resolution_2026-10-23/report.md)：三个 MC 差异已用精确参考和原流独立分类定位，六个 QC 尾部已输出有限 log 概率。原 float 和历史结果保留；以下说明对应新增 log 接口。

## 操作与输入

输入是一个细胞或细胞矩阵的**完整实测 Gene Expression 面板**、对齐基因名及独立分子保留率。`retention` 可以是 scalar，也可以是与基因对齐的向量。要求非负有限整数 counts、唯一基因名和全部 11 个规则 marker；未测基因不能补零。面板完整性由调用方依据原始数据确认，接口不能自动识别缺失的非 marker 基因。

分类条件包括 MS4A1 AND (CD19 OR CD79A)、CD3D/CD3E 均阳性且合计≥3、CD8A≥2、CD4阳性、CD14/CD68、MZB1/TNFRSF17，以及原规则中的谱系冲突和 CD4/CD8 双阳性排除。返回类别顺序为 B、CD8、CD4、myeloid 的候选标签，具体字符串见 `classes`。

QC 默认要求 retained total counts≥50、retained detected genes≥20。marker 分类、总数和检出数共享保留后的分子，不能用 `P(label)*P(QC)` 替代 `P(label AND QC)`。实现按六个互不重叠的 marker 组构造非负生成函数，与背景基因的 counts×detected 截断分布组合；同组规则与 QC 的依赖保留。

**图、节点数 N、边数 M 和 baseline eligibility 固定。** QC 失败、冲突和未决类别归入 OTHER；未删节点或重建图。原 baseline 不合格节点在批量接口中保持 OTHER，不能因下采样而重新进入分析。删除失败节点并改变分母是另一种操作，本接口不计算该操作。

```python
from spatial_membership_audit.candidate_qc import (
    candidate_joint_qc, candidate_joint_probabilities,
)

# counts 是完整实测面板的一个细胞；genes 与 counts 对齐。
cell = candidate_joint_qc(counts, genes, 0.25)
joint_class_probabilities = cell["probabilities"]
qc_failure_probability = cell["p_qc_failure"]
qc_failure_state = cell["qc_failure_status"]
qc_failure_log10 = cell["qc_failure_log10"]  # float 下溢时仍可为有限负数

# counts_matrix 行为细胞；baseline_eligible 是与细胞对齐的 bool mask。
batch = candidate_joint_probabilities(
    counts_matrix, genes, baseline_eligible, 0.25,
)
# batch["probabilities"] 为 N×4，可接固定图概率接口。
```

## 直接补事件与状态

`p_qc_direct` 和 `p_qc_failure_direct` 分别保存直接累加的 pass/fail；失败状态是不交的 `total<50` 或 `total>=50 AND detected<20`。最终 `p_qc` / `p_qc_failure` 使用较小一侧的直接事件概率和补事件恒等式，避免从已舍入到 1 的 pass 中相减恢复极小失败概率。`qc_evaluation` 记录求值路径；类别概率没有裁剪或归一化。

每类概率、QC pass 和 QC fail 均有支持状态：

| 状态 | 含义 |
|---|---|
| `STRUCTURAL_ZERO` | 根据原整数 counts、保留率及规则判定事件不可达 |
| `NUMERICAL_UNDERFLOW` | 事件可达，但 binary64 输出为零 |
| `POSITIVE_SUBNORMAL` | 正的非正规浮点数 |
| `POSITIVE_FINITE` | 正的普通有限浮点数 |

`p_qc=1` 可以伴随严格正的 `p_qc_failure`，因此不表示模型中的数学必然事件。批量接口在可达类别概率或 QC pass 下溢时拒绝继续交给下游，要求保留并检查逐细胞状态；QC failure 下溢仍单独标记，不能解释为无风险。baseline 不合格行的有效类别/QC pass 被置为零；其 raw direct 字段保留该行的假设下采样诊断，不应解释为 baseline 纳入概率。

实现使用 binary64，质量残差守卫为 2e-12；这不是任意输入或极小尾部的相对误差证书。尤其可达类别本身舍入或下溢时，不能仅凭下游浮点图模型的结构零宣称原始分子事件不可能。

## QC failure 的 log 输出

QC failure 为 `NUMERICAL_UNDERFLOW` 或 `POSITIVE_SUBNORMAL` 时，接口自动调用 `qc_tail.qc_failure_log_probability`，直接累加不交事件 `detected<min_detected` 或 `detected>=min_detected AND total<min_total` 的 log 概率。没有从接近1的pass相减，也不删除微小项。该辅助接口可直接用于完整counts面板的scalar/vector保留率。

| 新字段 | 含义 |
|---|---|
| `qc_failure_log` | 自然对数 ln(P(QC failure))；结构零为 `None`，数学上对应−∞ |
| `qc_failure_log10` | 十进制对数；不能通过对旧float=0取log重建 |
| `qc_failure_log_status` | `POSITIVE_LOG_FINITE` 表示可达事件有有限log，或 `STRUCTURAL_ZERO` |
| `qc_failure_log_method` | log DP、普通正float直接取log、原始支持或baseline不合格覆盖 |

float 输出和原支持状态不因 log 恢复而被篡改。例如约7.72e-694的风险仍是float=0、`NUMERICAL_UNDERFLOW`，同时输出log10约−693.112403。baseline不合格行的effective failure=1，log=0；其raw direct字段仍是未应用baseline门的诊断。新增的是QC failure的表示，类别概率下溢仍由原批量guard拒绝。

log DP也采用binary64，没有任意输入的严格相对误差界；本轮六项实际尾部均用独立精确有理数核验，最大log10绝对误差1.906e-12。若数值结果超出其非正有限域，接口明确报错，不静默裁剪。

## 实际验证范围

- 当前 17 项针对性检查通过：6 项新 log 检查、7 项完整规则测试、4 项四 marker 回归。包含完整小计数 Fraction 枚举、标量/向量、低 QC 门槛、无 QC 退化、p=1 hard labels、冲突、mask、非法输入和小概率状态；未重跑整个包的回归集。
- 两平台各按 cell_id 固定取 8 个历史合格细胞；CosMx 1000 基因、Xenium 377 基因，p=.25/.5/.75 共 48 个 cell×p；仅 p=.25 每细胞 512 次分子 MC，合计 8192 次。
- 原 QC pass 累加在 15/48 条略超 1，修复前代码和失败保留。首个实际故障细胞的失败风险为约 1.33284e-32，精确整数生成函数/有理数参考核验相对误差 4.13e-15。初版参考上界过宽而失败的记录也保留。
- 原80项MC比较中9个边界区间另存算术更正，其中7个错误排除得到修正；3个实际点位区间外差异保留。后续三项精确联合参考最大相对误差2.50e-15，8192原流独立分类全部一致，支持这三个固定实例的有限MC偏差归因；未追加估计抽样。原6个QC fail float下溢均已有有限log并获精确参考支持，48条旧字段不变。真实子集没有正CD8/CD4概率，仍不能声称其真实面板性能已覆盖。

上述证据支持指定 rarefaction 操作的有限实现核验。候选标签身份准确性、完整区域的联合 QC proximity、患者组间收益、普遍速度优势和真实物理噪声校准均不由本轮建立。
