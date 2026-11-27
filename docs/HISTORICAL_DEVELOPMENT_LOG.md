# Historical Development Log (`0.2.0` -> `2.0.0rc2`)

2026-11-01 新增[逐基因双向匹配](../DEPTH_MATCHING.md)：每基因只稀释观测均值较高的一侧，并提供 beta-binomial 直方图敏感性接口。原四对切片的配对内零比例/方差改善在固定 rho 网格保持，固定校准参数迁移仍未通过联合要求。均值相等是拟合恒等式；过离散接口尚未覆盖空间 P+、O/E、CV 或假阳性率。旧运行及函数默认行为保留。

2026-11-01 D 数值更正：修复混合类别节点在快速路径条件不满足时被漏算的问题；旧 `RESOLVED` 也可能受影响。E5 的1,510条异常和另外1,221条受影响旧成功记录单独复核，原运行保留。使用范围、修复机制及历史结果边界见[直接CV说明](../DIRECT_CV.md)。

冻结逐细胞分类器入口：[AnnData完整教程](../tutorials/frozen_classifier_anndata.md)与[可运行脚本](../examples/frozen_classifier_anndata.py)。教程使用合成计数和预先固定的线性模型，演示MC来源与误差边界、基因特异保留率、自定义callable和保存读回；不下载或训练模型。

2026-11-01 E4 接口修复：AnnData结果逐保留率保留分类器MC来源、draws/seed及误差适用范围，抽样零保留为未解，不再据此标为结构不可能；保留原图引擎的条件状态，整体MC输出不冒充端到端认证。单个基因特异p向量可运行并保存基因顺序，`p_grid`明确仅接受标量序列。字段与保存位置见[空间诊断指南](../ACTIONABLE_GUIDE.md)。本轮是局部工程修复，未发布新版本。

2026-10-31 修复与重评：[空间诊断指南](../ACTIONABLE_GUIDE.md)明确默认分界值仅为报告约定。R 使用共享基因联合状态概率及正确的条件联合矩，ratio CV 保留一阶 Delta 近似标签；SpatialData 请求的坐标转换失败时明确报错；自定义分类器须显式确认逐细胞假设，有限检查不构成独立性证明。D 释放不再使用的大数组并恢复逐阶计时，固定全片六配对已重建原始运行记录。Python 标准对象接口有本地测试；本机未安装 GiottoClass，原生 R/Giotto 宿主验收未运行。1,728 条既有轨迹的诊断分类不构成门槛或生物学效用验证。本轮改变了源代码，未发布新版本。

2026-10-28 新增实验性[Auto-Routing](../AUTO_ROUTING.md)：显式区分条件数值包围与近似筛查，支持D、scrambled Sobol及超过Sobol维度限制时的MC。K阈值仅为可配置启发式，采样置信度尚未校准；不会将采样结果标成D证书。原接口和历史证据保持，未验证组合性能或发布新版本。

2026-10-26 新增[完整端点扰动分布](../ENDPOINT_DISTRIBUTION.md)：小状态精确联合枚举＋大规模完整经验轨迹，覆盖O/E、两种置换分数、co-occurrence、K/L（none/translation）和双向G；36项检查、228项实际原生小例对照通过。20组真实输入保留576份分布、104,064条端点状态及六FOV的2,160条缺几何K/L。嵌套精确置换另限每尺度15边及状态预算；经验近似不认证稀有尾部。

2026-10-26 新增[逐原生输出通用核验](../NATIVE_VALIDATION.md)：InSituType/SingleR的372条真实存档标签端点通过逐实现ID对齐与独立复算；RCTD在固定128 spots上实际完成基线和两次半保留率重拟合，3幅连续组成场通过原生适配和R/Python独立核对。41项工程回归检查通过。原18组存档计划中6个BooleanRule组因源文件缺失保留为NOT_RUN；Cell2location/Tangram仍只有合成契约证据。支持三工具两语义的技术接入，未建立分类准确率或解析预测性能。

2026-10-23 新增[同图观察端点映射](../ENDPOINT_BRIDGE.md)：O/E、固定类别数置换z、端点归一co-occurrence和未校正K/L。co-occurrence/R映射穷举5,184个小图标签配置，K/L另有坐标例，本机Squidpy 1.6.5四半径实际核验保留原生零与无支持状态差别。不是这些端点的全套rarefaction分布传播，也不更改原CLI默认行为。

2026-10-23 新增[区域完整11标记联合QC接口](../REGIONAL_QC.md)：直接补事件log已进入区域NA，固定图输出条件O/E矩与直接CV。两个历史区域8条件保留肺结构性NA与肾低可用率；21项针对性检查、512分子draw、整数尾部及另写的边对参考均有完整记录。MC不足以核验两项肾CV；证书范围仍是输入浮点条件模型，未开展新队列或发布。

2026-10-23 后续三项 MC 差异与六项尾部修复：3 个解析联合概率与精确有理数参考一致，8192 原抽样独立分类重放一致；6 个 QC failure 下溢新增有限 ln/log10 输出并逐项核验。17 项针对性检查通过、48 条旧字段不变，原失败和float下溢状态保留。[接口字段](../FULL_RULE_QC.md)。未新增估计抽样或队列分析。

2026-10-23 新增[完整 11 marker 联合 QC](../FULL_RULE_QC.md)：支持实际候选规则、完整面板和 scalar/vector 保留率，固定图中 QC 失败归 OTHER。最终 11 项针对性检查通过；48 个 cell×p 与 8192 次分子抽样保留 3 个实际点位区间外差异和 6 个 QC 尾部下溢。首个真实故障的直接 QC 失败概率约 1.33e-32，精确有理数核验相对误差 4.13e-15。未发布新版本或运行新患者队列。

2026-10-23 条件CV预算修复：新增Cervilla完整11-marker规则小试的**5条Xenium未决已全部获证**，最大因子1.001061≤1.25，五条完整worker耗时3.911–4.998秒（上限8秒），核心计算最大3.090秒（上限6秒）。固定12条件为10正CV获证、2精确零、0未决；原失败、首条CosMx汇总故障及其缺失遥测保留。当前实现采用复合正权积分和有误差界的非负求和；本版本未重跑原72条肺/肾记录，两个版本的覆盖证据分别保留。未发布新版本。

2026-10-23 新增 [联合 QC 与 O/E→置换 z 开发接口](../JOINT_QC.md)：四 marker 联合 QC 经精确小例与两个历史窗口的分子 MC 核验；原 fixed-QC 路线及历史版本未改写。

**原肺/肾工作负载的历史全覆盖（v2实现）**：[直接条件比值CV](../DIRECT_CV.md)以`estimate_cv`输出数值包围和唯一主值。v2在同72条肺/肾记录的66条正CV均获≤1.25倍证书，原6条超时全部解决，肺/肾各33/33；单条最大4.178秒，保持原8秒硬限。旧60/66结果及6条失败完整保留，另6条结构性CV不可用也保留。这是固定条件模型/输入的数值覆盖。

2026-10-19开发增量：[原生重拟合审计](../REFIT_AUDIT.md)已完成既有InSituType/SingleR真实标签在六个FOV的空间端点复算。这是历史数据中的条件经验稳定性证据，未建立解析预测性能、真实连续场性能或身份准确性。

**早期 0.2.0 源代码开发增量**：[主流空间统计接口](../ECOSYSTEM.md)已增加 Squidpy、Giotto、cross-type Ripley K/L、标签与显式概率输入。原生小规模核验与形式/数值测试详见该文档。

## 2026-10-26 增量价值开发检查

新增[固定增量检查接口](../INCREMENTAL_VALUE.md)：半收缩保留率、训练区标准化、固定惩罚拟合、整组重采样和联合增益门。

## 2026-11-22 Rarefaction-Extrapolation (MITM), Conditional Coverage Matching & Joint-Permutation DUA (2.0.0rc2)

- **Unsupervised depth & conditional coverage calibration**:
  - `spatial_membership_audit.depth_matching`: `marginal_depth_ratio` and `select_auto_q` choose threshold-free common-depth thinning rates from post-QC positive transcript medians (`medpos`), while preserving legacy `balance_threshold < 1.0` when explicitly requested.
  - `spatial_membership_audit.coverage_matching`: `conditional_instability_metrics`, `split_half_disagreement`, `build_split_half_curve`, and `match_split_half_coverage` compute classifier instability strictly conditional on classifiable cells (preventing `UNRESOLVED`/zero-cell dilution while reporting `unresolved_frac` separately) and match split-half label coverage across heterogeneous panels and platforms.
- **Within-run Rarefaction-Extrapolation (`spatial_membership_audit.extrapolation`)**: `summarize_thinning_ladder`, `select_mitm_target_depth`, `build_extrapolation_guardrail`, and `extrapolate_ladder` extrapolate colocalization trajectories along the `1/sqrt(d)` shot-noise coordinate to meet-in-the-middle (MITM) or higher reference depths using hierarchical Empirical Bayes slope pooling (`prior_only`, `hier_eb`, `shrunk_step`), with automatic low-depth / high-flip guardrails falling back to rarefaction.
- **Detection-uncertainty-aware ensemble inference (`spatial_membership_audit.dua_inference`)**: `summarize_dua_draws`, `exact_perm_z_from_draws`, `joint_perm_pairs_from_draws`, `decompose_dua_uncertainty`, and `conformal_prediction_band` separate fixed-input Monte Carlo precision (`mc_se_logr`), valid-zero ratio expectations (`r_mean`, `n_defined`, `n_defined_z`), joint-permutation null calibration (`z_dua`, `null_sd_z`, `permutation_p_z`, `null_corr_z`), and out-of-sample split-conformal repeat prediction bands.
- **Thinning-stability weighting (`spatial_membership_audit.stability_weighting`)**: `cell_stability_weights`, `exact_weighted_perm_pair`, and `stability_weighted_colocalization` compute closed-form weighted permutation moments for unthinned native maps.
