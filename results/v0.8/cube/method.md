## Cube 方法在 Matmul 和 Addmm 的实例

输入采用连续 BF16 ND 的 A[M,K]、B[K,N] 和输出 Y[M,N]。Addmm 的 self 是 BF16 `[N]`，alpha=beta=1，cubeMathType=0。A5 已有三个 Addmm 小 shape 在单个 AIC 入口 `entry_kernel_2` 完成。预测假设每个输出 tile 在这条路径内读取并广播对应列 bias。属性或广播模式改变时需要重新识别实际调用链。

1. 用实现的切块规则求输出任务。初始块为 16×64，当任务数超过32时，在加倍行块或列块的候选中选择任务数最接近32的划分，每维上限256。令 `m_t,n_t` 为所选行列块尺寸和真实行列宽度取小值，`K_a=16 ceil(K/16)` 是计算对齐长度。总任务数 `P=ceil(M/t_m)ceil(N/t_n)`，波数 `w=ceil(P/32)`。这一步给出临界核任务量的估计。
2. 基本输入搬运工作为 `R=2(m_t+n_t)K_a/1024` KiB，BF16的2是每元素字节数。Addmm再加 `2 n_t/1024` KiB。Cube工作 `C=2m_t n_t K_a/131072`，单位为131072 FLOPs。FIXP输出工作 `F=2m_t n_t/1024` KiB。每个阶段 `s_j=a_j+b_j x_j`，x依次取R、C或F，a是该阶段固定微秒数，b是每工作单位的微秒数。这三组系数沿用v0.2 BF16 Matmul阶段数据。把各阶段工作分别代入，可检验跨算子复用的误差。
3. 首波填充 `D=s_R+s_C+s_F`，稳态间隔 `J=max(s_R,s_C,s_F)`。总流水时间 `B=D+(w-1)J`。Matmul继续使用已冻结的v0.6预测，对B、活跃核数及算法分支作原有控制校正。Addmm取 `T=H+B`，H是旧单tile Addmm trace内不被任何搬运/Cube/FIXP有效区间覆盖的时间。具体选取两个完整旧样本，将服务队列区间取并后从跨核总窗口扣除，H取中位数2.07636µs。曾用于v0.7验证的 `[9,64,128]` 在v0.8转为开发数据。H也包含未归因等待，各队列服务成本仍可能含共享资源效应。跨资源实际部分重叠和多核尾差会影响估计。

以 `[9,128,256]` 为例，输出按列分成两个16×64块，在两个核上同一波并行，每个核计算9×64有效输出。Matmul和Addmm都读取9×256的A片和256×64的B片。Addmm每块额外读64个bias，按128字节计入搬运，用同一块的输出列广播后写回。增加K会改变单块搬运/计算服务，增加M或N可能改变核数、块大小或轮次，这两种shape变化对耗时的影响不同。

验证先固定模型、二进制、调用合同与10个未见shape。每个case只使用shape和attrs得到预测，测量未命中预测路径的单次墙钟开销。CA计时取完整跨核指令窗口，单独检查全部输出与CPU参考、SoC身份和DB完整性。工作量从512到67,108,864个乘加项，覆盖列/归约尾块、单token、Q9、粗tile和大矩阵。KDA chunk由状态依赖和Cube/Vector协作驱动，独立运行完整64-token块验证其CA计时资格。

KDA chunk另测具有不同源码执行条件的两条路径。普通整块样例为 `[1,64,1,1,128,128]`，在预激活gate下复现CA重复set_flag错误。随后用独立调用器验证 `[1,64,2,2,128,128]`。BF16、FP32 raw gate、A_log、safeGate及useGateInKernel为真、偶数value heads、S按64对齐且关闭qg/v_new/h导出，对应固定参考源码的内联Gate和Post-WU/FwdH融合候选。CPU按每个head单独计算 `gamma=lowerBound/(1+exp(-exp(A_log)*g))`，并以gamma驱动串行状态递推和chunk内三角矩阵求逆，检查全部请求的O、最终状态、gk、Aqk、Akk与输入状态完整性。此次CA在117.8秒发出三个重复set_flag错误，主kernel没有完整时间线，因此缺少该实现的可信时延。两个合同各预设10形状，先行样例的失败保留，其余未运行。冻结样例及接口参数见`safe-chunk-freeze.json`。实际已编译分支的源码映射仍待tid/tiling/PC证据核实。

KDA chunk的可执行结构入口为`operator_model.cube_v8_chunk.predict('kda_chunk',shape,attrs)`。设单个序列切成长度为c_i的块，块上限C，块数n=ceil(S/C)，并行状态链数U=B×Hv。对单条链，有效因果元素对数P=Σc_i(c_i+1)/2，块矩阵稠密大小D=Σc_i²。QK与严格下三角KK工作分别为2KP与2K(P-S)，Post-WU为2(K+V)D，FwdH为4SKV次FLOPs和nKV次状态衰减乘法，Finalize为2SKV+2VD；全算子乘U。K/V是每头key/value宽度。FwdH在同一链内按块顺序推进，QK/KK/Finalize可按不同块和头分核。该入口返回各阶段工作、依赖、分块、适用属性与尚缺的三角求解/资源成本。上述两次v0.8实测均遇同步错误，故可执行结构模型的时延字段为null。

独立留出中，Addmm `[512,512,256]` 预测6.4022 µs、参考4.4788 µs。最后结束核的搬运、Cube、FIXP实测区间并长2.1206 µs；该核MTE2单独占1.2648 µs、FIXP占1.9545 µs，两者存在重叠。模型填充段把三项服务串加为4.3258 µs。窗口内未被这些资源覆盖的实测时间为2.1067 µs，与控制项2.0764 µs接近。该shape的主要待修正项为首tile内的阶段重叠及实际分块。观测来源及阶段明细见[临界核诊断](worst-addmm.json)。

### Chunk KDA 的十组独立结构核对

从两份已冻结清单选取十个不同 shape，B=1，S 从 1 到 256，Hq=Hv 取 1、2、4、8，K=V 取 16 或 128，C=64。CPU 核查器枚举每个头、块、有效 token 和有序 token 对，检查六个阶段的逻辑工作量、工作为零的阶段、DAG 顺序、任务数、串行步数和状态字节。S=1 时严格下三角为空，KK 工作为零，该节点在 DAG 中省略。用非零初始状态和每头一个活跃 K/V 通道，进一步对照块内三角求解和逐 token 递推的输出、各块末状态；已激活 gate 和 raw-safe gate 均覆盖。十组均通过，FP64 运算的最大差值为 9.72×10⁻¹⁷。该结果限于逻辑计数与稀疏数学恒等式。安装包 CA 耗时仍为 0/10，BF16 全输出、内部融合与物理吞吐待验证。逐项结果见 [chunk-structure-validation.json](chunk-structure-validation.json)，可运行 `python3 scripts/cube-v8-chunk-structure.py` 复现。
