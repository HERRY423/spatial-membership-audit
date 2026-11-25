# 空间比较诊断与标准对象接口

本指南解释固定分类规则、固定空间图与指定分子保留模型下的可用率和条件离散度。默认分界值是可修改的报告约定，尚未验证为自动筛选门槛；它们不提供 p 值、FWER/FDR 控制或生物学效应证据。

## 两个需要同时报告的量

- `P_plus`：在互斥细胞类别模型下，两个类别都有至少一个细胞的概率。每个细胞只能属于一个类别；同一细胞的两个类别概率不能相乘后计作共同出现。
- `CV(R | E>0)`：比值在有定义的实现中的相对离散度。低 CV 可以与低 P₊ 同时出现；前者仍是有效的条件描述，不能据此前推比较通常有定义。

当图无边时，O/E 不可定义。图有边、两类可同时出现而跨类接触恒为零时，R=0 是合法值，但相对 CV 不可定义。模型中的零接触不能直接解释为生物学排斥。

## 诊断范围与行动

| 状态 | 默认范围 | 可采取的行动 |
|---|---|---|
| `TIER_1_ROBUST` | P₊≥0.95，CV≤0.25 | 报告指定操作下的高可用、低条件离散；跨样本推断仍按样本设计完成。 |
| `TIER_2_MODERATE_DISPERSION` | P₊≥0.95，0.25<CV≤0.50 | 并列报告条件均值、CV 及其计算方法；避免只展示一个随机抽稀实现。 |
| `TIER_3_CONTACT_VOLATILE` | P₊≥0.95，CV>0.50 | 核查接触计数和门控边界；不能从 CV 单独推断关键细胞数量或波动原因。 |
| `TIER_4_AVAILABILITY_WARNING` | 0.50≤P₊<0.95 | 显示不可定义概率 1−P₊；改变 ROI、半径或门槛时另标敏感性分析。 |
| `TIER_5A_CONDITIONING_TRAP` | 0<P₊<0.50，CV≤0.50 | 同时报告低条件离散和频繁失定义，不能仅按 CV 排为稳健。 |
| `TIER_5B_SEVERE_FRAGILITY` | 0<P₊<0.50，CV>0.50 | 同时报告可用率低与条件波动大；不能把浅深度失定义解释为生物学缺失。 |
| `STRUCTURAL_ABSENT_POPULATION` | P₊=0 | 保留 NA；检查类别支持、规则与 marker 覆盖。极小正概率不得改写为零。 |
| `STRUCTURAL_NO_GRAPH_EDGES` | M=0 | 核对坐标单位、预设建图方式；比值保留 NA。 |
| `DEFINED_ZERO_CONTACT` | P₊>0、条件接触均值=0、M>0 | 报告 R=0、P₊ 与 CV 未定义；排斥需另有合适的零假设比较。 |
| `NUMERICAL_UNRESOLVED` | 输入或数值状态未解决 | 保留状态；若采用 MC/Sobol，标明近似与抽样误差。 |
| `NON_MONOTONIC_VETO_ENTRY` | 保留率曲线非单调 | 检查完整曲线和规则转变；否决基因脱落是待核查的可能原因。 |

扩大 ROI、改变邻域半径、合并视野或删去类别会改变目标量或样本选择。应预设这些选择，或完整报告原规格与修改规格；不能调到“通过诊断门槛”为止。细胞、类别对、保留率和随机种子都不会增加供体/切片重复数。数值包围区间也不是生物学置信区间。

## Python：AnnData、SpatialData 与 Squidpy

冻结分类器的完整可运行入口见[AnnData教程](tutorials/frozen_classifier_anndata.md)及[配套脚本](examples/frozen_classifier_anndata.py)，包含固定模型、标量/向量保留率、MC边界、预期拒绝示例与AnnData保存读回。

```python
from spatial_membership_audit.standard_ecosystem import audit_anndata, audit_spatialdata

result = audit_anndata(
    adata, p=0.5, rule="common125_candidate_v1",
    count_layer="counts", region_key="fov",
    graph_method="radius", radius_um=20.0, coordinate_unit="um",
    cv_mode="direct", inplace=False,
)

result = audit_spatialdata(
    sdata, table_key="table", shapes_key="cells",
    coordinate_system="global", p=0.5,
    rule="common125_candidate_v1", graph_method="radius",
    radius_um=20.0, coordinate_unit="um", inplace=False,
)
```

这里的 20 µm 是演示参数，实际应采用研究预设值。`coordinate_system` 必须是 shapes 中已登记的目标系统；转换失败会报错，不会继续使用原坐标。接口核对实例标识、唯一性和有限质心；`global` 等名称不证明物理单位是微米，用户须根据数据来源给出单位。已有邻接图的语义须单独核对，坐标转换本身不会重建已有图。

自定义 marker 规则使用 `marker_rule_from_dict` 或 `MarkerRuleSpec`。共享基因的谓词由联合状态计算；状态数过大应显式失败，不能宣称任意复杂规则都可廉价计算。冻结逐细胞分类器在 0<p<1 时使用 Monte Carlo 估计分类概率；后续 D 的数值区间以这些输入概率为条件，不覆盖上游 MC 误差。自定义 callable 须在确认语义后显式传入 `assume_cellwise=True`（AnnData 接口为 `classifier_assume_cellwise=True`）；内置冻结线性分类器按其定义处理。有限的顺序/子集检查不是节点独立性的证明；分类器须固定参数且不使用跨细胞重拟合、归一化或平滑。

分类器概率的来源信息保留在每一行，并同时写入 `result.attrs['provenance']['probability_runs']`；使用 `inplace=True` 时，另保存在 `adata.uns['spatial_membership_audit']['provenance']['probability_runs']`。按 `retention_key` 对应每个保留率，避免把整条曲线误记成一个种子：第 i 个点的实际种子为 `seed + i * 10007`。

| 字段 | 含义 |
|---|---|
| `semantics`、`probability_method` | 确定性端点分类或分子抽稀 MC；不是分类器原始 posterior |
| `n_draws`、`seed` | 实际抽样次数和该保留率的种子；p=0/1 的确定性分支为0次，记录的seed未用于抽样 |
| `independence_basis` | 内置逐细胞定义，或调用方确认的假设；不证明任意外部模型独立 |
| `upstream_probability_uncertainty`、`cv_interval_scope` | 明确下游CV数值区间是否以MC点估计为条件；不包含上游MC误差，不是端到端置信区间 |
| `p_finite_is_estimate` | MC输入得到的可用率是代入估计值 |
| `conditional_cv_status`、`conditional_guarantee` | 保留图引擎对给定概率矩阵的原始状态与保证 |
| `guarantee` | MC输入的整体输出至多为 `MC_PLUGIN_ONLY`；未解时为 `NONE`，不冒充端到端精确认证 |

MC未抽中类别或接触时，保留估计零以及原图引擎状态，但整体 `cv_status=UNRESOLVED_UPSTREAM_SAMPLING`、`probability_support_status=UNRESOLVED_MC_SUPPORT`，归入未解状态；不能据有限抽样断言事件在模型中不可能。确切的p=0/1零状态和无图边状态仍按已知结构处理。本接口没有新增上游MC置信区间或完整误差传播认证。

`p` 支持一个与 `adata.var_names` 严格同序的基因特异向量；返回行的 `p` 用 `vector:[...]` 保留全部数值，provenance同时保存 `values` 和 `genes`。该输入没有自动定义的标量深度顺序，所以 `retention_curve_status=NOT_EVALUATED_NONSCALAR_RETENTION`，不计算标量曲线非单调性。`p_grid` 仅支持标量序列，嵌套向量会明确拒绝；多个基因特异方案应分别调用并保留各自的provenance。

`audit_squidpy_nhood` 可并列显示 Squidpy 邻域置换统计与抽稀诊断。二者的零模型不同，P₊/CV 不能自动校准置换 z 值或控制其多重检验错误率。是否有推断增益须另作公平基线比较。

## R：精确可用率、条件 Delta 近似与 Giotto 接口

```r
source("R/standard_ecosystem.R", encoding="UTF-8")
probs <- sma_evaluate_custom_rule_r(
  counts, genes, eligible=fixed_mask, p=0.5,
  rule_spec=sma_stromal_myeloid_rule_spec_r()
)
mom <- sma_pair_availability_and_moments_r(
  probs, adjacency, col_a="stromal", col_b="myeloid"
)
```

R 规则格式为 `gene`、`gene_sum`、`all_of`、`any_of`、`not`，与 Python 字典格式不同。R 使用 capped binomial 联合状态计算共享基因与 veto 的概率，并检查类别互斥；默认最多 65,536 个联合状态，超出报错。只在固定 eligibility 和独立分子保留假设下解释这些概率。

R 的 `P_plus`、接触均值/方差和条件计数联合矩按互斥节点模型计算。`cond_cv_ratio` **是使用条件联合矩的一阶 Delta 近似**，不是 Direct Engine D；结果保留 `ratio_cv_method`、`ratio_cv_is_exact=FALSE` 和 `moment_status`。相对 CV 在零均值时为 NA。

`sma_audit_giotto_object` 支持原生 Giotto 提取路径和列表输入。原生 Giotto 的端到端验收须有真实对象运行记录，列表测试不能替代它。预先提供的概率矩阵绑定一个保留率；接口拒绝把同一矩阵用于多个 p 值并伪装成保留率曲线。

## 如何解释已有案例

每个案例须绑定来源、切片/区域、类别对、保留率、图和规则版本。同一 `g27_14` 区域的 CD4–myeloid 与 CD8–CD4 是不同端点，不能混用其 R、P₊ 和 CV。已有轨迹被分到某一诊断类别，只证明分类规则运行过，不验证诊断门槛或行动效用。回顾性模型内表现与独立生物学验证应分别报告。
