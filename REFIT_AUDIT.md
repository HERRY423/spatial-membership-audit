# 原生重新注释结果的空间稳定性审计

**2026-10-26增量**：[新逐实现入口](NATIVE_VALIDATION.md)支持每次输出独立的节点/类别轴和独立端点参考，旧API保持兼容。真实证据现覆盖InSituType/SingleR的372条存档标签端点及RCTD实际3次原生组成场；原计划6个BooleanRule组因源文件缺失本轮未运行。下文“连续场仅有工程合同测试”保留为2026-10-19的历史状态，当前覆盖以[新报告](../../review/classifier_general_validation_2026-10-26/report.md)为准。

`refit_audit.audit_refits` 在显式执行清单下接收完整标签或连续场。它不运行分类器，不把分类器 posterior 当作抽稀重分类概率，也不使用独立节点解析公式。每次计数扰动对应一个完整空间输出；共用参考、cohort 频率重估或空间平滑产生的跨节点依赖保留在该输出内。

真实执行证据见 [2026-10-19 报告](../../review/native_annotation_bridge_2026-10-19/report.md)。当前真实验证限于既有 InSituType/SingleR **硬标签**；Cell2location、RCTD、Tangram 的连续场仍仅有工程合同测试。此路径增加真实注释流程的经验端点评估能力，不提供解析预测精度或加速保证。

```python
from spatial_membership_audit.refit_audit import audit_refits

result = audit_refits(
    baseline=full_count_labels,
    realizations={'0': labels_after_refit_0, '1': labels_after_refit_1},
    failures={'2': 'native fit did not converge'},
    planned_ids=['0', '1', '2'],
    node_ids=cell_ids, graph=fixed_graph,
    classes=['B', 'CD8', 'OTHER', 'undefined'], pair=('B', 'CD8'),
    provenance='saved native output identifiers and count-perturbation manifest',
)
```

每条成功结果必须覆盖同一固定节点集合。成功与失败 ID 必须严格分割完整计划；缺失、重复、遗漏不默默忽略。`finite_fraction_of_planned` 把失败留在分母；`finite_fraction_given_success` 另报成功条件下的可计算率。拟合失败、成功但 E=0、E>0 且 O=0 分开。有限次全 NA 不能证明结构性不可能；基线 NA 也不意味着所有抽稀结果必然 NA。

`mean_R_finite`、`sd_R_finite`、`cv_R_finite` 均以有限结果为条件，CV=样本SD/均值；单条有限结果不计算SD/CV，均值零不计算CV。`mcse_mean_R_finite` 还要求各完整扰动实现条件独立同分布；对链相关 posterior draws 或重复复制的结果不可作为有效 MCSE。它不是 CV 的区间或生物学区间。

对于连续输出指定 `kind='fields'` 和 `value_semantics='cell_abundance'|'composition'|'expected_mapped_mass'`。每次 refit 只传一个预先指定的场汇总；若同时传播 within-fit posterior，必须另建嵌套层，不能展平成额外计数扰动。场端点是节点间共丰度 O/E，不能称单细胞接触或身份概率。原生 rejects/缺失 spot 不可静默删去、填0或改成概率1；需预先指定节点级缺失合同，当前严格接口会拒绝不完整输入。

若要评估**预测**，应使用独立扰动批次：从固定校准批次估计所需分布或代理，再在新抽稀、原生重拟合结果上评价误差、可计算率和成本；不能用同一批实现既估计又声称验证。固定单细胞推断与整队列重拟合须分别标识，后者不能因边际频率可求而自动进入 `pair_moments`。
