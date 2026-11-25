# 主流空间统计接口：开发增量

2026-10-18后续：[完整概率、联合标签与连续场接入](POSTERIOR_BRIDGE.md)增加真实InSituType保存结果导入，以及Cell2location/RCTD/Tangram不同输出语义的开发适配。后三者仍只完成合成合同测试；不能称真实native生态性能验证。[SSc第二规则正式执行报告](../../review/scope_transfer_recovery_2026-10-18/report.md)保留全部36条结构性NA及其原始计数支持证据。

本增量将已有 O/E 解析器扩展到 Squidpy 邻域富集、Giotto 空间邻近分数，以及 cross-type Ripley K/L。
代码位于 `src/spatial_membership_audit/ecosystem.py` 和 `R/ecosystem.R`。这是当前源代码的开发接口，未修改旧 wheel、历史手稿或正式分析归档，也未发布新版本。

## 能力与证据

|接口|当前能力|核验边界|
|---|---|---|
|Squidpy `nhood_enrichment`|逐 region 调用官方函数；支持 1.6 tuple/1.8 named result；保留 count、Z、非有限状态、种子、版本与源码 hash|1.6.5/1.8.3 的12细胞合成夹具；3种无向图和有向 kNN；不代表真实数据外部验证|
|Giotto `cellProximityEnrichment`|R 原生调用；显式单 region；原生模拟与结果完整保留；区分缺类别与工具未返回；Python 可导出原生边表|4.2.3 实际原生执行；半径、kNN、Delaunay 三种输入图；图由 SciPy 构建，不冒称 Giotto 默认图|
|spatstat `Kcross`|R 原生桥接；Python 提供相同窗口下的直接估计与固定位置分类不确定性传播|矩形窗口、none/translation 两种校正；9个半径的 K/L 原生数值对照；不含任意组织掩膜/isotropic/inhomogeneous K|
|InSituType|原生 `clust` 导出与完整细胞 ID 对齐|已有真实原生 RDS 的导入读回；本轮没有重新分类|
|SingleR|`labels` / `pruned.labels` 导出；pruned NA → 显式 UNKNOWN|原生 S4 输出格式测试和已有真实标签表导入；本轮没有重新分类|
|CellTypist|消费 `AnnotationResult.predicted_labels`，兼容 Categorical；majority voting 列须显式选择|1.7.1、已有本地 Immune_All_Low 模型、4个合成细胞的原生 annotate 与ID对齐通过；真实数据比较未运行|

核验脚本和回执在 `../../review/spatial_ecosystem_2026-10-15/`。全部新运行均为小规模工程核验，正式真实数据验证另定范围。

## Python：从 AnnData 到原生 Squidpy

```python
from spatial_membership_audit.ecosystem import graph_from_anndata, native_squidpy

graph = graph_from_anndata(
    adata, region_key='region_id', adjacency_key='spatial_connectivities',
    directed=False, coordinate_unit='um',
)
result = native_squidpy(
    graph, adata.obs['candidate'].fillna('UNKNOWN').astype(str),
    n_perms=100, seed=1729,
    categories=['B', 'CD8', 'OTHER', 'UNKNOWN'],
)
```

需要现有 NumPy/SciPy/pandas/AnnData/Squidpy 环境。函数不下载、不安装、不自动训练，不修改传入 AnnData。
固定 cell ID、明确 region、微米单位、有向性必须由调用方指定。不同 region 分开计算。重复 ID、自环、跨 region 边均拒绝；不把权重悄悄二值化，不把有向 kNN 默默改成无向图。
`native_squidpy` 的 `categories` 可包含本轮缺失类型，以保留 `ABSENT_TYPE`；原始非有限 Z 仍保存，不用零代替。

需要新图时使用 `build_graph(xy_um, cell_ids, regions, method=...)`：

- `radius`：显式 `radius_um`。
- `knn`：显式 `k` 和 `knn_symmetry='directed'/'union'/'mutual'`；每 region 至少 k+1 个点。
- `delaunay`：未经距离剪枝的 SciPy Delaunay；退化几何直接失败，不自动抖动坐标。

并列距离的 kNN 邻居可能受底层实现影响，正式分析应冻结实际邻接矩阵。图构建选择不等于各工具默认设置。

## 同一图交给 Giotto

```python
from spatial_membership_audit.ecosystem import giotto_edge_table
giotto_edge_table(graph, xy_um).to_csv('network.csv', index=False)
```

导出的唯一无向边包含 `from/to`、坐标、`distance` 和 `weight`。Giotto 需要这些字段；有向图和非二值图拒绝导出。R 端以 `createSpatNetObj` 和 `setSpatialNetwork` 放入已经建立的单 region Giotto 对象，随后：

```r
source('R/ecosystem.R')
result <- sma_giotto(
  gobject, cell_ids, labels=setNames(labels,cell_ids),
  network_name='supplied_network', n_simulations=100L, seed=1729L,
  region_ids=setNames(rep('region1',length(cell_ids)),cell_ids),
  pairs=list(c('B','CD8'))
)
```

返回 `raw` 包含未变换的原生 `raw_sim_table/enrichm_res`；`pair_states` 保留缺失/未返回/非有限。
调用方为零模型提供的说明默认为 `UNVERIFIED_NATIVE_IMPLEMENTATION`。本轮仅对实际 Giotto 4.2.3 实现逐函数匹配了已保存源码，因此可以确认其边端点标签多重集置换机制。升级版本后须重新核对，不能仅凭文档沿用。

## cross-type Ripley K/L

```python
from spatial_membership_audit.ecosystem import cross_k, predict_cross_k
observed = cross_k(
    xy_um, labels, ('B','CD8'), [0,10,20,30],
    window=(0,400,0,400), correction='translation',
)
# P: 同一细胞顺序下、互斥类别的条件概率；此处不是原始分类器分数。
predicted = predict_cross_k(
    P, xy_um, [0,10,20,30], a=0, b=1,
    window=(0,400,0,400), correction='translation', independent_nodes=True,
)
```

`window` 必须是明确的观察窗口，不能用细胞坐标极值冒充真实采集边界。只接受窗口内点和不超过窗口内切圆半径的 r。
R 入口 `sma_kcross` 使用官方 `spatstat.explore::Kcross`；其半径网格须从0开始、等距递增。
K=0 是可计算的零；任一类别缺失时 K/L=NA。预测返回精确分子/分母矩与协方差，以及 K/L 的 delta 近似；K=0 处 L 的导数奇异，方差保留 NA。没有自动生成显著性或置信包络。

## 注释输入与概率不能混淆

`align_annotation` 读取任何方法的标签表并严格按 ID 对齐；全细胞集合必须一致，不允许删去拒判细胞。`from_celltypist` 消费原生结果；R 中 `sma_export_singler` 和 `sma_export_insitutype` 导出同一标签合同。

CellTypist 1.7.1 的本地模型路径应传 `Path(model_path).as_posix()` 给 `Model.load`：其路径判断依赖 `/`，Windows反斜杠可能误触发模型目录联网。核验入口固定已有模型、不使用默认模型名称下载。示例的合成表达仅检验接口，不检验模型准确性或正式面板适用性。

`categorical_input` 是单独的概率入口：需要明确 `semantics`、来源和 `independent_nodes=True`；完整类别概率必须行和为1，绝不自动归一化、softmax 或裁剪。

- SingleR `scores` 是相关性分数，`delta.next` 也不是概率。
- CellTypist one-vs-rest 分数/概率不能直接视为和为1的互斥类别分布；majority voting 还可能引入细胞间依赖。
- InSituType `prob` 只表示已分配类别的值，不能仅凭该向量恢复完整多类别不确定性矩阵。
- 后验不确定性和分子抽稀后重新分类的概率是不同随机对象。即使行和为1，也不自动证明概率校准或细胞独立。
- 对重拟合、共同参考估计、空间平滑等带来的细胞间依赖，保留整条联合标签轨迹，逐轨迹计算原生端点；不能只用边际概率运行独立节点闭式公式后声称精确。

`weighted_pair_moments` 包含节点内的 `diag(p)-p pᵀ` 和共享节点引起的边依赖；不包含未知跨节点分类相关。零模型置换方差与分子抽稀技术方差是不同输出。

## 解释规则

1. Squidpy 原生 Z 使用有限置换的均值和总体标准差（ddof=0）；无向图额外给出的 `exact_node_null` 是固定类别数量节点置换的精确矩，不替代原生值。对有向图不套用无向解析式。
2. Giotto 本机版本原生 enrichment 为 `log2((original+1)/(simulations+1))`，不是未加伪计数的 O/E。`giotto_endpoint_null_moments` 单独提供该已核实端点置换的精确矩。
3. K 的 `πr²` 是齐次独立类型点过程参考；固定坐标随机标签置换是另一零模型。技术扰动敏感性不等于生物学伪阳性率。
4. 不将跨零 Z/enrichment/L−r 的 CV 用作稳定性指标；已有 `summarize_trajectory(..., nonnegative_ratio=False)` 报告 SD、正负/零和非有限计数。
5. 当前没有通用 `E0<5` 可靠性阈值，也没有“对所有分类器均成立”的实证结论。

## 官方参考（2026-10-15 核对）

- [Squidpy API](https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.nhood_enrichment.html) 与 [原生源码](https://squidpy.readthedocs.io/en/stable/_modules/squidpy/gr/_nhood.html)
- [Giotto API](https://giottosuite.readthedocs.io/en/latest/subsections/md_rst/cellProximityEnrichment.html)；本机源码优先于概括性文档措辞。
- [spatstat Kcross](https://search.r-project.org/CRAN/refmans/spatstat.explore/html/Kcross.html)
- [SingleR](https://bioconductor.org/packages/release/bioc/vignettes/SingleR/inst/doc/SingleR.html)、[CellTypist](https://celltypist.readthedocs.io/en/latest/celltypist.annotate.html)、[InSituType](https://github.com/Nanostring-Biostats/InSituType)
