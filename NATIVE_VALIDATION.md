# 逐原生输出的通用接入核验

2026-10-26。真实技术接入覆盖 InSituType、SingleR 的完整硬标签及 RCTD 的连续组成场。[报告与完整结果](../../review/classifier_general_validation_2026-10-26/report.md)。Cell2location/Tangram仍只有合成契约测试。

`native_validation` 为每次输出分别接收节点与类别轴，解决共用一份 `node_ids` 不能表达逐次重排的问题。每条记录含 `node_ids`、`classes`、`values`、`semantics`、`provenance`。节点和数值列只依据ID对齐，不丢点、不补类、不取argmax、不裁剪、不自动归一化。

```python
from spatial_membership_audit.native_validation import audit_native_outputs

# 每条记录的ID轴必须来自该条实际输出。
baseline = dict(node_ids=baseline_ids, classes=declared_classes,
    values=baseline_labels, semantics='hard_labels',
    provenance='saved native baseline and source hash')
refit_1 = dict(node_ids=refit_1_ids, classes=declared_classes,
    values=refit_1_labels, semantics='hard_labels',
    provenance='saved whole native refit 1 and source hash')
result = audit_native_outputs(
    baseline=baseline, outputs={'r1': refit_1},
    failed={'r2': 'native fit did not converge'},
    not_run={'r3': 'not reached after stop'},
    planned_ids=['r1', 'r2', 'r3'], graph=fixed_graph,
    classes=declared_classes, pair=('B', 'CD8'),
    semantics='hard_labels', provenance='input, graph and run manifest',
)
```

`outputs`、`failed`、`not_run` 必须无交集且完整覆盖计划。非法输出为 `INVALID_OUTPUT`，拟合失败为 `FIT_FAILED`，未运行是 `NOT_RUN`；有效输出的E=0为 `NO_FINITE_RATIO`，E>0且O=0为 `FINITE_ZERO`。基线非法时仍保留其他有效输出，但不生成delta。有限比例分别保留计划分母和有效输出分母；有限次全NA不能证明结构性不可能。

图必须固定、单region、二元无向。每条输出须覆盖完整节点和声明类别。硬标签某一类别可以没有成员，但声明类别全集不能静默缩小。`UNKNOWN`可预先显式声明，不能拿它隐式填补丢失节点。

连续场是node×class矩阵，语义为 `composition`、`cell_abundance` 或 `expected_mapped_mass`。RCTD先由原生 `normalize_weights` 处理raw weights，经 `rctd_composition_field` 取得已归一化的点估计；接口仅检查行和。每次refit只传一幅预先指定的场，posterior draws需要另设嵌套设计，不能冒充额外refits。单场不产生SD/CV。

每条端点与独立边列表O和直接去对角节点对求和E核对。连续参考计算为O(N²)，适用于有限规模核验，不承诺大图加速。rtol=1e-10、atol=1e-12只是算术一致性门，不是浮点误差证书、身份准确率或科学效能保证。整幅输出保留节点依赖，不把posterior当作thinning类别转移概率。

|方法|本轮实际证据|范围|
|---|---|---|
|InSituType|62个真实RDS新读回；6 FOV、186端点通过|同一历史library；未新增拟合|
|SingleR|原生流程保存预测表；6 FOV、186端点通过|不声称另有62个SingleR RDS|
|RCTD|真实保留面板fixture，3次实际原生拟合，128×11完整输出|单来源、两个抽稀场不足以评估SD/CV精度|
|BooleanRule旧对照|历史结果保留；6组/186计划端点本轮NOT_RUN|原轨迹和cells.tsv缺失，未换用不同哈希文件|
|Cell2location/Tangram|既有合成契约|本轮无真实原生联合draw/映射证据|

41项相关工程回归检查通过；375条当前核验原生端点通过确定性轴重排，72个故障派生输入被拒绝。重排和故障注入不增加原生实现数或生物学重复。旧 `refit_audit.audit_refits` API保持兼容；新入口适合每条输出有独立ID轴的流程。
