# Experimental Auto-Routing: D and stochastic screening

2026-10-28。新增显式路由API/模块CLI，原`primary_cv.estimate_cv`及旧CLI不变，未发布版本。它是工程原型，不是新的已验证SOTA方法；主文和历史交付不变。

## 定位与路由规则

用户先指定结果要求，软件再选引擎。D提供给定概率/固定图/算术前提下的数值包围；采样提供近似点估计。选择依据和所有尝试写入`route_trace`、`attempts`，不静默替换保证。

| 优先级 | 条件 | 行为 |
|---|---|---|
| 1 | 逻辑支持上没有有限R，或没有可能的正接触 | 返回相应结构状态；不把它写成资源耗尽 |
| 2 | `requirement="enclosure"` | D，失败保留未决；禁止随机降级 |
| 3 | screening且`factor<=strict_factor_hint`（默认1.01） | 优先D；这只是工程选择，不保证本次预算成功 |
| 4 | screening且`K_union<=direct_support_hint`（默认1000） | 优先D；阈值可配且未经性能验证，不是通用复杂度边界 |
| 5 | 其他screening请求 | 默认scrambled Sobol；超过当前实现维度上限时改用MC并记录原因；也可显式选择MC |

`K_union`是A或B概率严格大于0的节点并集，包含正支持孤立节点；另列`K_A/K_B`。不按“小概率”截断。它不等于总细胞数，不能把文献中未定义清楚的K直接套入。

稀有接触诊断为`draws_per_replicate * E[O] < 1`的浮点启发式，单独给warning。它不从全零pilot样本断定真实均值为0，也不强制把巨大K送入D。既有importance sampling可处理部分稀有接触，因此不声称D独占；本路由版本**尚未集成M1**。低可用性、稀有分母、浮点概率分辨率都可能造成额外问题，该诊断并非完整检测器。

实际检查的本机SciPy1.16.3 `qmc.Sobol.MAXDIM=21201`。一活跃节点一坐标的当前实现不能直接对几十万K运行Sobol；超过限制时记录`SOBOL_DIMENSION_LIMIT`并转MC，不分块拼接后冒称原高维Sobol。这里未实施全切片吞吐量验证。

## 两类输出不可混写

- D成功：`guarantee=CONDITIONAL_NUMERICAL_ENCLOSURE`、`cv_interval`、`factor_achieved`。证书条件限于输入模型与算术前提，不包含概率估计误差、物理检测误差或形式机器证明。
- 采样成功：`guarantee=APPROXIMATE_POINT_ESTIMATE`；`cv_interval=null`，`factor_target_assessed=false`，`coverage_calibration=NOT_ESTABLISHED`。factor只参与选路，不代表采样已达到该精度。
- 多个独立scramble/MC stream给出完整replicate CV及描述性min/max。它们**不是置信区间，也不是覆盖率校准**；CV是非线性比值估计，样本均值为零/极稀有时尤其不能赋予伪置信度。
- 输出逐draw状态计数：`finite`、`unavailable`、`finite_zero`、`positive`，每replicate也分别保存。正常返回结果保留所有计划replicate身份与已执行数，包括资源拒绝及预算耗尽；输入验证异常和进程崩溃不属于正常返回契约。超时未执行不会被计成科学失败。
- `SAMPLE_NO_FINITE_OBSERVATIONS`、`SAMPLE_NO_POSITIVE_CONTACTS`、结构性`NO_FINITE_RATIO`、`ZERO_NUMERATOR_CV_UNDEFINED`及`BUDGET_EXHAUSTED`严格区分。采样CV恰好为0只是一项点估计，不升级为`EXACT_ZERO_CV`。

“无方差”的准确含义仅是D没有随机种子造成的Monte Carlo变动，不能指R的模型方差为0，也不能指没有数值/模型误差。“无状态丢失的数学闭包”不是本软件保证：D会未决，采样会漏掉稀有事件，均必须显式报告。

## 预算、内存与降级

`max_seconds`是调用内包含验证和准备的软时间目标，不含模块导入/CLI文件读取，不是操作系统硬限。检查发生在阶段/批次边界，故可能超出；输出`within_soft_budget`和实际耗时。生产硬限制需要外部worker，当前未集成。

采样用有限batch，估算每batch工作数组不超过`max_batch_bytes`；这不是总进程内存保证，图、概率、Sobol方向表/初始化、稀疏乘法和Python开销另有内存成本。不能据此称为工业级性能。

只有screening显式设置`allow_approximate_fallback=True`时，D才先用剩余软预算的一半，并在普通UNRESOLVED后用剩余预算尝试采样。原D输出完整保留。数值冲突等异常不被采样结果掩盖；enclosure模式不允许这一选项。

## 使用

```python
from spatial_membership_audit.auto_routing import route_cv

# 有条件数值包围要求；不允许以随机估计填充失败结果。
bounded = route_cv(p, graph, requirement="enclosure",
                   accuracy_factor=1.01, max_seconds=6)

# 接受近似筛查；0禁用“小支持优先D”启发式。
screen = route_cv(p, graph, requirement="screening",
                  direct_support_hint=0, accuracy_factor=1.25,
                  draws=1024, replicates=4, seed=17, max_seconds=6)
```

```text
python -m spatial_membership_audit.auto_routing --probabilities p.npy --adjacency g.npz --requirement screening --support-hint 0 --draws 1024 --replicates 4 --seed 17 --out routing.json
```

输出文件已存在时拒绝覆盖。所有方法使用原N/M定义E，压缩支持后仍保留正支持孤立节点。采样按完整计划每scramble的2幂draw数生成序列；预算中止的未完整scramble不声称平衡性质。使用ddof=0的经验CV；不能不加说明与历史ddof=1结果混合。

## 当前证据与待检验问题

合成工程检查覆盖保证不降级、独立小例枚举、原N/M/孤立节点、MC标量参考、结构态与抽样零、维度回退、预算/内存返回、种子和CLI。它们不证明路由比always-MC/Sobol更快、更准或更有用。旧D/Sobol benchmark使用不同实现/调优版本，不能直接搬来作为这个组合原型的速度证据。

要主打一个新组合方法，后续必须在未用于选路调参的输入中，以同一目标、同预算和完整开销比较auto、always-D、always-MC/Sobol与M1，并报告路由错误成本、失败率及适用规模；若需要CI，还须独立验证覆盖。若auto只是给常规任务增加准备/错误选路开销，应改路由或不宣称增量。此检验完整成本UNKNOWN，本轮未运行，不扩展既有科学预算。
