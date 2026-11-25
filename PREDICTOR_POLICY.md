# Delta / Expected-O 的可选决策接口

当前主工作流使用[直接条件比值CV](DIRECT_CV.md)的`estimate_cv`；本页保留历史两个近似的诊断/风险接口。主输出不再由Delta或Expected-O二选一产生，也不把本页的相对候选风险界当逐记录准确度保证。

新增接口位于 `spatial_membership_audit.predictor_policy`。不改变既有
`pair_moments` / `pair_moments_prepared` 或冻结稿件的输出；不自动启动MC、
exact求解、阈值搜索或预算追加。本接口尚未获得独立新来源的实用性能验证。

```python
from spatial_membership_audit.analytic import pair_moments
from spatial_membership_audit.predictor_policy import assess_moment_output

moments = pair_moments(probabilities, adjacency)
decision = assess_moment_output(
    moments,
    regret_factor=1.10,   # 示例偏好：相对两候选中最佳者，不是相对真值
    accuracy_factor=1.25 # 逐记录绝对误差要求，不是集合平均F
)
```

调用者应在查看目标前选定容差；没有适用于所有科学问题的默认容差。

| 返回状态 | 允许解读 | 不允许解读 |
|---|---|---|
| `RELATIVE_REGRET_ONLY` | 几何折中值相对两候选最佳者的额外乘法误差有界 | 已证明绝对准确，或已识别哪种候选更准 |
| `NEEDS_MODEL_REFERENCE` | 在指定选择风险容差下需模型参考或弃权 | 默认改选Expected-O/Delta |
| `MODEL_REFERENCE_RESOLVED` | 在有效同模型区间覆盖目标的条件下，参考中心达到指定逐记录因子 | 已校准真实实验技术误差 |
| `NONPOSITIVE_OR_UNAVAILABLE_CANDIDATE` | 正值log-loss规则不适用 | 将零、NA、下溢都改成很小正数 |
| 原结构状态 | 原样保留`NO_FINITE_RATIO`、`ZERO_NUMERATOR_CV_UNDEFINED`等 | 用模拟未见事件证明不存在事件 |

`ranking`与`accuracy_status`分开：更接近不等于足够准确。两个候选都可能失败，
但区间中心合格；反之，能证明排序也可能不能证明绝对误差合格。

v2的`compromise_regret_*`仅绑定`geometric_compromise`。如果推荐值改为参考区间
中心，`recommended_regret_factor_bound`置为None，不能拿原折中值的界担保新推荐。
极端正浮点输入的因子超出显示范围时返回None和`OUT_OF_FLOAT_RANGE`，仍保留
有限的log界及精确有理数阈值判断；None不表示数学界无限或已经达标。

`CVInterval`支持三种来源：

- `exact_enclosure`：调用者须提供同输入、同图、同保留率、同条件目标的有效上下界。
  数据类不会因标签名而验证其真实性。不得把旧评分真值冒充免费部署参考。
- `fixed_budget_hoeffding`：用`bounded_cv_interval`从预定固定预算iid样本计算，
  附明确alpha和支持上界。有限零保留，NaN单列；禁止边看边追加后重复使用同一alpha。
  多记录、多次查看须单独分配错误概率。它可能非常宽。
- `approximate_mc`：例如旧jackknife近似区间；只显示、不升级为严格排序或准确度认证。

评价目标必须设`reference_role="evaluation_only"`，不会参与部署选择。可接受的部署
参考还必须核对`evidence_id`指向的输入、目标及证书；字符串本身不构成证书。

本轮保存结果回放：720正CV中512条满足示例regret门槛，208需参考；6结构记录保留。
该覆盖不是准确度通过率。57组旧MC在保守有限样本界下全部未决。因此此接口是
可审计决策层，不是已经证明普遍更准的新预测器。

数学推导、回放证据与审查见
`review/predictor_policy_2026-10-19/`。原解析引擎及其验证状态独立于本接口；
不能用本模块测试通过来认证上游模型或原始数据。
