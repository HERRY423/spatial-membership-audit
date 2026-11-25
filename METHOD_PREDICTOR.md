# 解析支持量预测：开发接口

当前CV主输出见[直接条件比值CV](DIRECT_CV.md)。下文`pair_moments`仍提供具名Delta近似和精确支持矩；主入口`primary_cv.estimate_cv`另外直接计算比值矩包围，达到调用者精度后给唯一主值，失败保留未决。两个输出不混称。

新增analytic.py与analytic_adapters.py，不改0.2.0原有门控、CLI及冻结归档；未公开发布新版本。
NumPy/SciPy为基础依赖；AnnData适配器在现有AnnData环境使用，不自动安装依赖。

```python
from spatial_membership_audit.analytic_adapters import predict_anndata

rows = predict_anndata(
    adata, p=0.5, count_layer="counts", eligible_key="baseline_eligible",
    region_key="region_id", adjacency_key="spatial_connectivities",
)
```

counts必须是未经归一化的原始整数；显式固定baseline_eligible；图为用户已按微米构建的二值无向图，禁止自环与跨region边。不要把空间坐标本身当图，或默默把加权图二值化。必须测得全部11个marker，缺基因直接拒绝。返回每region六个互斥候选类型对的矩、可计算概率、delta CV及警告。

直接矩接口：

```python
from spatial_membership_audit.analytic import marker_probabilities, pair_moments

prob = marker_probabilities(counts, genes, baseline_eligible, p=0.5)
result = pair_moments(prob, adjacency, a=0, b=1)
```

p可为[0,1]内标量或与genes顺序一致的逐基因保留率；q>1拒绝，不裁剪。逐基因场景支持是计算能力，不是实验校准。类别顺序在CLASSES中。原非候选可以因否决信号丢失而进入，所以prob覆盖所有固定节点。

- mu_O/mu_E、var_O/var_E、cov_O_E：指定独立删除模型下的无条件矩。
- *_finite：E>0条件下的矩；p_finite单列，不把不可计算当零。
- ratio_of_means不是精确E[O/E]；mean_R_delta2和cv_R_delta为不同阶数的delta近似。
- p_finite=0或无图边时比值NA；有E且O恒为0时R=0、CV=NA。
- approximation_warning提示稀疏/可变分母风险，阈值是诊断标记，不能当验证准确性证书或据此删除失败。

signed Squidpy z-score或Giotto enrichment等保存输出用summarize_trajectory(values, nonnegative_ratio=False)报告SD、正负和NA；不跨零计算CV。该函数不调用官方工具，也不生成p值。分类器posterior不能直接作为本模型类别概率；重拟合或空间平滑会引入细胞依赖，须额外建模。

推导、全部开发结果、反例与后续合同见[Method开发包](../../review/method_route_2026-10-15/report.md)。支持的真实数据开发范围为4输入、3保留率、20um、既存30轨迹；任意p是接口/数学范围，并非所有p都有真实数据验证。
