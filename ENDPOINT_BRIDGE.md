2026-10-26：下文为历史观察值桥接；[新增完整分布接口](ENDPOINT_DISTRIBUTION.md)已实现既定端点的小状态精确/大规模完整经验传播，保留输入条件和缺几何状态。

# 同一观察状态的空间统计量映射

`endpoint_bridge.observed_endpoint_bridge(graph, labels, ('A','B'))` 接受单区域 `SpatialGraph` 和完整硬标签向量；拒绝有向、加权、自环图以及跨区域混合。返回 O/E、固定类别数节点置换 z 和 endpoint-normalized co-occurrence。可选 `window_area` 以 μm² 给出真实观察窗面积；这同时要求调用方确认图为相应半径的未校正图，接口无法从邻接矩阵证明其几何来源。需要坐标检查或 translation correction 时用 `ecosystem.cross_k`。

固定图的 N 个节点、M 条无向边，A/B互斥，O为跨类边数，d_A、d_B为两类节点的度数总和：

- `R = O / [2M n_A n_B / (N(N−1))]`。
- `z = (O−E)/sqrt(Var_perm O)`：使用固定类别数节点置换精确矩；不等于有限置换软件输出的逐位值。
- `C_AB = 2M O/(d_A d_B)`：与相同距离对集合、全体类别下Squidpy co-occurrence的归一化代数对应。它不是以细胞比例代替边端点比例。
- `C_AB/R = (2M)^2 n_A n_B/[N(N−1)d_A d_B]`，有定义时成立；同一个R不确定C_AB。
- 未校正同一半径图：`K_AB = area*O/(n_A n_B) = area*2M/[N(N−1)]*R`；`L_AB=sqrt(K_AB/pi)`。有类型但无边时K=0，R仍可未定义。

这些关系逐条用于已观察标签或完整重拟合轨迹。不能把E[R]代入非线性公式当E[z]、E[C]或E[L]。固定正几何因子使未校正K与R在同一可用域的CV相同；translation-corrected K的权重和co-occurrence的随机度数分母不满足该结论。对z、L−r等跨零分数报告SD、符号和NA，不报告CV。

OTHER、UNKNOWN和QC失败保留在固定图的总端点质量中，不删节点。无A/B边端点支持时 `co_occurrence=None, NO_ENDPOINT_SUPPORT`；原生Squidpy可能输出0，二者状态必须一并保留。完整图中R=1而z因零null方差未定义；co-occurrence可为N/(N−1)，并不必须为1。

测试覆盖4节点64图×81三类别标签组合、相同R不同C的反例、孤立类型、真零/无定义、完整图、空图和坐标K/L。原生小例核验状态见[本轮记录](../../review/issues_4_9_resolution_2026-10-23/native_check.json)，不是新真实数据验证。更高层的原生注释仍须保留整条联合标签轨迹；此接口不拟合、不预测标签、不将posterior解释成thinning概率。

依据：[Squidpy官方源码](https://squidpy.readthedocs.io/en/stable/_modules/squidpy/gr/_ppatterns.html)，核对日期2026-10-23；在线1.8.3与本地1.6.5应分别记录。特别是重合坐标、浮点距离阈值及split实现，只有实际pair support相同时才能使用对应关系。本轮不更改原CLI默认行为。
