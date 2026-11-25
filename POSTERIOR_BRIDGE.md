# 概率、联合标签与连续空间场接入

**2026-10-26当前状态**：[逐实现通用核验](NATIVE_VALIDATION.md)新增真实RCTD连续组成场证据：128固定spots上实际完成3次原生full-mode拟合，与独立R/Python端点复算一致；InSituType/SingleR共372条真实存档标签端点通过新接口。Cell2location/Tangram仍限合成契约。下文2026-10-18/19关于“RCTD未执行”的文字为当时状态，当前覆盖见[新报告](../../review/classifier_general_validation_2026-10-26/report.md)。

**2026-10-19后续进展**：[原生重拟合审计](REFIT_AUDIT.md)已将真实存档InSituType/SingleR hard-label接入六个FOV的空间端点评估。下文2026-10-18导入检查状态作为历史记录保留；Cell2location/RCTD/Tangram的真实连续场验证仍未执行。新结果见[报告](../../review/native_annotation_bridge_2026-10-19/report.md)，不外推为解析预测有效性。

开发源码增量，2026-10-18（America/Los_Angeles）。不覆盖历史稿件、wheel或ZIP，不表示发布完成。

原来的 `pair_moments` 已接受连续的类别概率；限制来自随机对象和跨细胞独立假设，而非“只接受0/1”。新增 `posterior_bridge.py` 处理已有联合实现和连续空间场。后者是单独的空间统计端点，不把原11-marker的预测精度转移给它。

| 输入 | 接入方式 | 本轮核验与边界 |
|---|---|---|
| 完整互斥单细胞类别概率 | `ecosystem.categorical_input` → `pair_moments`，调用者明确独立节点假设 | 既有接口；行和1不证明校准或独立，posterior不等于thinning重分类概率 |
| InSituType原生结果 | R `sma_export_insitutype_posterior` 合并retained/lost logliks，用原生转换函数导出完整类别表 | 两个已有真实baseline RDS导入读回，18,768细胞×10类；未重拟合、未作空间预测或新验证 |
| 共同重拟合后的完整标签轨迹 | `joint_label_summary`，每行一条完整联合标签实现 | 不需要节点独立；合成依赖反例、节点对齐、有效零/NA测试；本轮未新增真实分类器端点 |
| Cell2location原生 `w_sf` 完整抽样数组 | `cell2location_field_samples` → `continuous_field_summary` | 只验合成输入合同；需显式draw×spot×type，保留同一个联合draw。均值/分位数不可冒充draw，不能独立拼batch后声称保留全局相关 |
| RCTD已归一化weights | `rctd_composition_field` → `continuous_field_summary` | 只验合成输入合同；是spot比例的点估计，SD/CV为空，不产生posterior不确定性 |
| Tangram `mode='cells'` 的cell×spot矩阵 | `tangram_cell_mapping_field`，按完整source cell标签累加为spot×type映射质量 | 只验合成输入合同；不是物理细胞数，不支持clusters/constrained的隐式权重或过滤 |

在2026-10-18本节记录形成时，后三者没有运行原生工具/真实来源验证；当时的可执行导入和端点路径还不是生态迁移实证。未知来源、缺节点或不合法概率直接拒绝；不补零、不自动裁剪、归一化或重拟合。

## 联合标签轨迹

```python
from spatial_membership_audit.posterior_bridge import joint_label_summary
result = joint_label_summary(
    saved_labels, cell_ids, graph,
    classes=['A','B','OTHER','UNKNOWN'], pair=('A','B'),
    realization_ids=saved_draw_ids, mechanism='refit_technical_labels',
    provenance='source file hash, native tool/version, refit design and draw IDs',
)
```

`graph` 是已有 `ecosystem.SpatialGraph`，必须是单region、固定节点、二元无向图。`saved_labels` 为draw×cell；不依据边际概率重新抽样。UNKNOWN作为显式类别保留节点。接口返回逐实现O/E和有限率、有限域均值/SD/CV；E=0时R为None，E>0而O=0时保留R=0。单条实现没有方差估计。失败的原生拟合应在上游完整清单记录，不可从draw集合无声剔除；本函数拒绝缺失标签，不能从已传入子集自动发现上游删掉的整条draw。

## 连续空间场端点

对一个spot/节点场中的两类非负值a、b，定义

`O = aᵀ G b`

`E = [2M / (N(N−1))] × Σ(i≠j) aᵢ bⱼ`

此E是将整个节点的多类向量共同均匀置换时O的期望，保留同一spot内两类共存并排除same-node项。测试枚举了3节点全部6个置换；one-hot时退化为原categorical端点。它描述节点间丰度/组成关联，不提供细胞接触数、显著性检验或因果结论，也不将posterior波动解释为分子抽稀波动。

```python
from spatial_membership_audit.posterior_bridge import (
    cell2location_field_samples, continuous_field_summary,
)
payload = cell2location_field_samples(
    native_w_sf_draws, sample_ids=draw_ids, spot_ids=spot_ids,
    class_names=cell_types,
    joint_draw_provenance='saved full joint draw export, source hash and sampling procedure',
)
result = continuous_field_summary(graph=spot_graph, pair=('Fibroblast','Myeloid'), **payload)
```

`posterior_joint_fields` 只表示调用者声明的样本机制，不证明MCMC混合、变分近似精度、联合抽样正确性或概率校准。无自动置信区间；finite_fraction是所供实现的经验比例。多region必须分开计算，不能把它们的N/M混合归一化。

## InSituType完整概率导出

```r
source('R/ecosystem.R')
posterior <- sma_export_insitutype_posterior(
  native_result, cell_ids, expected_classes=c(reference_classes,'undefined'))
```

`prob` 是所选类的概率，不能恢复其他类；`logliks` 还可能移走没有MAP调用的类别。新函数要求同时提供 `logliks_from_lost_celltypes` 和完整期望类别清单，不删除 `undefined`。利用已安装InSituType自己的 `logliks2probs` 转换，保留native四位小数logliks造成的误差。共同估计cohort频率带来的依赖仍然存在，导出成功不授权使用独立节点公式声称精确posterior传播。

## 官方来源核对

2026-10-18以官方文档和源码检索/读取得到；检索词仅含公开工具名、输出字段与接口术语，无未公开数据或稿件。来源会更新，本地原生导入使用InSituType 2.0。

- [Cell2location输出是cell abundance](https://cell2location.readthedocs.io/en/latest/cell2location.html)
- [RCTD输出结构与归一化weights](https://github.com/dmcable/spacexr/blob/master/documentation/README.md)
- [Tangram cell-to-spot映射返回值](https://tangram-sc.readthedocs.io/en/latest/classes/tangram.mapping_utils.map_cells_to_space.html)
- [InSituType保留与lost类别logliks源码](https://github.com/Nanostring-Biostats/InSituType/blob/main/R/insitutypeML.R)
- [InSituType关于过度自信的说明](https://github.com/Nanostring-Biostats/InSituType/blob/main/FAQs.md)
