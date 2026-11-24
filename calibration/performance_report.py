"""Build the Chinese v1.13.1 report from the repository root with Matplotlib."""
import csv
import json
import re
from pathlib import Path
from dataclasses import replace

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt, font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from operator_model import estimate, load_chip

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs"
DATA = ROOT / "calibration/candidate_indexer"
OUT = DOC / "operator-sim-report-zh.pdf"
font_manager.fontManager.addfont("/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf")
mpl.rcParams.update({"font.family":["DejaVu Sans","Droid Sans Fallback"],"pdf.fonttype":42,
                     "axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
ink, teal, blue, amber, muted, light = "#182f41","#007d79","#4773a1","#b75923","#476474","#e9f1f2"
W,H = 595,842
pdf = PdfPages(OUT,metadata={"Title":"昇腾算子性能建模：方法、校准与系统外推","Author":"operator-sim","Subject":"v1.13.1 双芯片校准与系统评估"})
count = 0
fig = None
# Use the renderer's composite-font metrics for Chinese and Latin text alike.
def width(s,size):
    return fig.canvas.get_renderer().get_text_width_height_descent(s,font_manager.FontProperties(size=size),False)[0]*72/fig.dpi

def txt(x,y,s,size=12,color=ink,bold=False):
    fig.text(x/W,1-y/H,s,fontsize=size,color=color,va="top",weight="bold" if bold else "normal")

def para(y,s,size=12,w=501,x=47,color=ink,leading=None):
    line=""
    for tok in re.findall(r"[A-Za-z0-9.,/%+×−_=⋅→<>·⁄-]+|.",s):
        if line and width(line+tok,size)>w:
            txt(x,y,line.rstrip(),size,color)
            y+=leading or size*1.62
            line=""
        line+=tok if line else tok.lstrip()
    if line: txt(x,y,line,size,color)
    return y+(leading or size*1.62)

def rect(x,y,w,h,c):
    fig.add_artist(Rectangle((x/W,1-(y+h)/H),w/W,h/H,transform=fig.transFigure,facecolor=c,edgecolor="none",zorder=0))
def page(ch,title,sub=""):
    global count,fig
    if fig is not None:
        pdf.savefig(fig);plt.close(fig)
    count+=1
    fig=plt.figure(figsize=(W/72,H/72),dpi=100)
    txt(47,31,"OPERATOR-SIM    /    性能研究报告",9,teal,True)
    txt(456,31,"v1.13.1   /   950",9,muted)
    rect(47,55,501,1,teal)
    txt(47,79,ch,11,teal,True);txt(47,106,title,24,ink,True)
    if sub: para(151,sub,11,color=muted)
    rect(47,804,501,.6,"#a9bdc0")
    txt(47,815,"昇腾算子性能建模    ·    2026 年 11 月",8,muted);txt(526,813,f"{count:02d}",10,teal,True)

def heading(y,s): txt(47,y,s,15,teal,True)
def note(y,s): para(y,s,9.1,color=muted,leading=13.5)
def panel(y,s):
    rect(47,y,501,48,light);para(y+10,s,13,w=477,x=59,leading=16)
def axis(x,y,w,h): return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def picture(file,y,h):
    a=axis(45,y,505,h);a.imshow(plt.imread(DOC/file));a.axis("off")
def table(y,columns,rows,widths,step=25,size=10):
    for idx,row in enumerate([columns]+list(rows)):
        if idx==0: rect(47,y-4,501,step,ink)
        elif idx%2: rect(47,y-4,501,step,"#eef4f4")
        x=48
        for v,col in zip(row,widths):
            fs=size
            while width(str(v),fs)>col-6 and fs>5.5:fs-=.25
            txt(x+2,y,str(v),fs,"white" if idx==0 else ink)
            x+=col
        y+=step
    return y
def source(s): note(767,"数据与复核  /  "+s)
def get(path):return json.loads(path.read_text())
summary=[]
for line in (DOC/'error-analysis.md').read_text().splitlines():
    if line.startswith('| [') and '](errors/' in line:
        cells=[v.strip() for v in line.strip('|').split('|')]
        if len(cells)==9: summary.append([re.search(r'\[(.*?)\]',cells[0])[1],*cells[1:]])
requests=list(csv.DictReader((DATA/'v113_context_requests.csv').open()))
page("摘要  /  ABSTRACT","昇腾算子性能建模","从单核指令流水，到长上下文推理的整图预测")
para(218,"系统设计者真正需要的是：给定一次单卡调用，换精度、换并行切分或换芯片后，哪一段会变慢？operator-sim 从实现使用的分块出发，计算每核要搬多少数据、执行多少运算，再按缓冲、发起间隔与依赖把阶段排成时间线。输出既有设备微秒，也有计算量、访存需求和活动核数。",14,leading=24)
heading(370,"校准结果")
para(403,"950DT 和 950PR 提供指令级 CA 参照。基础发布集中，30 项执行合同有双端可评分窗口，28 项达到当前门槛，其中 5 项通过双端冻结后的新形状验证。冻结独立样本和用于开发的旧窗分别计分。")
heading(513,"系统效果")
para(546,"在三个模型、33 组配置中，所有精细预测都满足同精度、同工作量的峰值下界。Flash 的 1Mi、单请求 prefill 末块由上一版的 480.550 ms 修订为 458.229 ms。完整末块仍属外推，实际验证来自 950 上更短的原生设备窗。")
heading(653,"最值得投入的地方")
para(684,"补足系统可见的前序状态；定位混合精度矩阵乘的长尾；约束未来芯片每核供数和 Vector 吞吐的不确定性。这三项分别解决输入不足、流水定价偏差和跨代假设。")
source("模型快照 75fd7ff；docs/error-analysis.md、docs/context-sweep.md。全文的 CA 指仿真器设备指令窗。")

page("01  /  理论","从理想下界走到完成时间","同一个算子有三种时钟：理想资源下界、设备执行时间、主机生成预测的墙钟。")
panel(210,"T 理想 = max（Cube 工作 / 峰值，Vector 工作 / 峰值，冷读写字节 / HBM 带宽）")
para(281,"同一执行引擎上的不同位宽分别除以各自峰值，再相加。计算量以 ops 计，乘加算两次；字节除以 B/s 得到秒。若只用一个固定效率，启动、尾波和串行的归约等待都会被压进同一个数。")
heading(390,"模型怎样确定关键路径")
for i,(t,s) in enumerate([("工作","从每卡张量、精度、布局和因果边界得到有用工作；填充槽及解包另计。"),("分配","用 UB、L1、L0 容量和核数选 tile，再把整块、尾块与拆分归并分给各核。"),("排程","每项操作等前驱完成和资源空闲后才启动。取最晚核的结束时间，再计必要的归并。")]):
    y=429+68*i;rect(47,y,57,40,light);txt(58,y+10,t,12,teal,True);para(y+2,s,11.4,w=428,x=117)
heading(648,"怎样读“等效利用率”")
para(679,"U = T 理想 / T 预测。系统 cold_io 对同一冷读写下界取最大值，故 U ≤ 100%。CA 指令窗里的 GM 读取可能命中 L2，输出也可能仍在片上。拿逻辑字节除以 HBM 峰值再除以这个短窗，可能超过 100%；缺少物理流量计数时，物理 HBM 利用率记为未知。",11)
source("docs/roofline-audit.md；docs/modeling-strategy.md；芯片峰值只用于对应的执行精度。")

page("02  /  VECTOR","量化为什么要看首读和写尾","DynamicMXQuant，BF16 输入 [9, 5120]，每 32 个数共享一个量化尺度。")
para(203,"46,080 个输入分成 1,440 组。读 92,160 B，写 46,080 B 的 MXFP8 和 1,440 B 的尺度。每行拆给五个 AIV，各处理 1,024 个元素。任务共占 45 核，整片峰值无法描述首块等数据的时间。")
picture("examples/assets/quant.png",297,285)
note(589,"图 1   另取 PR 多轮输入 [2048, 5120] 展示流水。模型 17.834 µs、CA 18.504 µs。下文回到九行的小输入，比较启动在总窗中的占比。")
para(638,"PR 模型的设备时间为派发 3.336 + UB 内流水 1.031 + 排空 0.010 = 4.377 µs，CA 为 5.025 µs（−12.90%）。同一输入在 DT 的预测 / CA 为 4.980 / 4.867 µs。RMSNorm 还需要行归约、缩放和输出写回，必须重新安排依赖。",11.6)
source("docs/examples/quant.md 与 assets/quant.json；保存的流水图仅用于该算例。")

page("03  /  CUBE","矩阵乘的耗时由最晚的波决定","BF16 MatMulV3，按 M、N、K 记为 [1280, 2048, 512]。")
para(194,"有用工作为 2MNK = 26.84 亿 ops。DT 上 256×256 的输出块共有 5×8=40 个。首波占 32 核，剩下八块各切四份。K 方向有两页，每页四条 MMAD，因此全片发出 512 条，与 CA 对得上。",11.6)
picture("examples/assets/matmul.png",278,337)
note(622,"图 2   同一设备零点下，比较模型关键路径与最后完成核。紫色是矩阵乘加，蓝绿色是数据搬运；第二页的读可以覆盖在上一页计算上。")
para(676,"DT 预测 12.449 µs，CA 12.219 µs（+1.89%）。PR 只有 28 个 AIC，首波后 12 个母块各拆为两份，实发 416 条 MMAD，预测 / CA 为 14.027 / 13.218 µs。跨芯片时分块、波数、每波服务率都重新计算。",11.3)
source("docs/examples/matmul.md、assets/matmul.json；实际核验按 MMAD 精度、几何和物理核计数。")

page("04  /  CUBE + VECTOR","融合注意力如何同时使用两种引擎","BF16 FA，单请求、单头、Q=9、D=64、KV=131,072。")
para(194,"每页 256 个 key，共 512 页。一个 AIC 做 QK 和 PV，两路 AIV 处理分数与归约。令 I 为稳态每页最慢的资源或指令发起间隔。总时间约为首轮准备 + 511I + 末页排空；把各引擎总时长直接相加会重复计算重叠段。",12)
picture("examples/assets/fa.png",285,275)
note(564,"图 3   950DT 的 128Ki 算例：预测 274.987 µs、CA 274.821 µs。这里的长循环受页发起节奏限制；头数、Q 长度和精度会改变可用并行。")
para(622,"本例 Cube、Vector、K/V 读、MTE1、FIX、控制需求分别为 356、375、819、464、178、853 周期/页。控制节拍来自旧 DT 128 页窗口中段 K/V 发起点的斜率。511×853.405 / 1650 = 264.30 µs，1650 是每微秒周期数。加首尾后 PR 为 274.988 µs，CA 为 265.893 µs。",11.6)
source("docs/examples/fa.md、assets/fa.json；页的工作量由实现决定，每核吞吐由目标配置决定。")

page("04  /  状态与组合","同样的 shape，还需要哪些输入","对归约、递推和按专家分组的工作，前一个调用与 token 的去向都可能改变下一步。")
heading(206,"mHC Post：把同一次 CA 分别交给两种入口")
para(243,"用芯片、完整形状和 CA 时延配对，两种入口一一对应 239 个记录。下面单列 token ≤16 且通道 ≥2048 的组，观察前序信息对小批宽通道的作用。该分组只用于分析误差。",11.7)
paired={}
for line in (ROOT/"calibration/reference/windows.jsonl").read_text().splitlines():
    r=json.loads(line)
    if r["contract"] in ("mhc_post · stream","mhc_post · shape"):
        key=(r["chip"],tuple(r["shape"]),r["ca_us"])
        pair=paired.setdefault(key,{})
        branch=r["contract"].split()[-1]
        assert branch not in pair
        pair[branch]=r["predicted_us"]
assert len(paired)==239 and all(len(x)==2 for x in paired.values())
vals=[]
for sku in ("950DT","950PR"):
    for wide in (True,False):
        sample=[(key[-1],pair) for key,pair in paired.items() if sku in key[0] and ((key[1][0]<=16 and key[1][-1]>=2048)==wide)]
        wapes=[100*sum(abs(pair[b]-t) for t,pair in sample)/sum(t for t,pair in sample) for b in ("shape","stream")]
        wins=sum(abs(v["stream"]-t)<abs(v["shape"]-t) for t,v in sample)
        vals.append([sku,"小批宽通道" if wide else "其他",len(sample),*[f"{v:.2f}%" for v in wapes],wins])
table(318,["芯片","分组","窗口数","仅 shape WAPE","有前序 WAPE","误差减小的窗"],vals,[64,99,60,93,96,89],step=24)
heading(474,"KDA：一条状态链上的 token 有先后顺序")
para(509,"单请求、三步、K=V=128 的 FP32 状态矩阵占 65,536 B。每步都依赖上一步写出的状态；同链按序排程，不同头或独立链才可并行。首次装入、每步更新和最终写回分别计价。",11.6)
heading(606,"MoE：先按专家数 token，再计算各次投影")
para(642,"8 个 token 分到两个专家时，[1,7] 和 [4,4] 总工作可以相同，尾块和并行度却可能不同。输入宽 256、中间宽 128 时，门控双投影加下投影合计 1,572,864 个 Cube ops。还需按实际精度计 SwiGLU、量化、中间结果写回，以及图上的专家交换。",11.6)
source("calibration/reference/windows.jsonl；docs/examples/kda_recurrent.md、moe_ffn.md。前序改善来自开发回放。")

page("05  /  跨芯片","换芯片时，哪些量能够带走","保留算法和依赖，按新芯片重新求并行度、分块和每阶段价格。")
table(202,["单卡资源","950DT","950PR","970DT 情景"],[
 ["AIC / AIV","32 / 64","28 / 56","120 / 120"],["频率（GHz）","1.65","1.65","1.80"],
 ["Cube 16 / 8 / 4 位（TOP/s）","432 / 865 / 1730","378 / 756 / 1513","1770 / 3540 / 14160"],
 ["Vector 16 位（TOP/s）","54","47","110.592（假设）"],
 ["HBM 容量 / 带宽（GB、TB/s）","103.08 / 4.0","120.26 / 1.4","288 / 14.4"],
 ["L2 容量","128 MiB","112 MiB","200 MB；30 TB/s"]],[182,100,100,119],step=24)
para(395,"表中 950 为独立算子内置配置；端到端另用系统提供的 950DT 预算。峰值 = 核数 × 每核每周期 MAC × 2 × 频率。芯片总峰值已包含核数和频率，按活动核折算时只能计一次。970DT 的 UB 和若干 MTE、控制周期沿用参考配置；实现是否仍然适用需另验。",11.5)
heading(485,"本报告复算：一次只将一种资源减半")
c=load_chip("Ascend970DT")
targets=[c,replace(c,hbm_bandwidth_gbs=c.hbm_bandwidth_gbs/2),
         replace(c,details=dict(c.details)|{"l2_bandwidth_gbs":15000}),
         replace(c,vector_tops={k:v/2 if v else v for k,v in c.vector_tops.items()})]
cases=[("矩阵乘","matmul",(1280,2048,512),{},"bf16"),("FA","fa",(9,131072,64,1,1),{},"bf16"),
       ("动态量化","quant",(9,5120),{},"bf16"),("Flash 末块单层","candidate_filtered_indexer",(1,4096,32,128,1048576,2048,512),{"implementation":"a5_qsli_projection","table_pages":8201},"mxfp4")]
sensitivity=[]
for name,op,shape,attrs,dtype in cases:
    sensitivity.append([name,*[f"{estimate(op,shape,attrs,chip=chip,dtype=dtype,memory_policy='cold_io').latency_us:,.2f}" for chip in targets]])
table(532,["预测，µs","原配置","HBM 带宽 ÷2","L2 带宽 ÷2","Vector 峰值 ÷2"],sensitivity,[128,73,100,98,102],step=24)
para(676,"FA 的 Vector 峰值减半后，耗时增加约 52%；矩阵乘则对 L2 带宽更敏感，重新搜索后活动核从 117 变为 112。相同报价表示这些算例的当前模型受其他约束；换 shape 或迁移实现仍需重算。",11.4)
source("src/operator_model/data/chips/；上述 970DT 数值全部为架构外推。GB、TB 为十进制。")

page("06  /  校准与误差","平均误差会掩盖什么","先核对实现与精度，再对齐物理核的计时窗口。只在边界一致的窗口上计算误差。")
para(197,"一次完整 CA 观察从最早核的首条设备指令到最后退休。先看输出正确性，再看库内核身份、指令次数、分核和结束事件。调过参数的样本计入开发回放。冻结模型后才取得或揭盲的时间，才可评价该版本在留出集上的表现。",11.8)
panel(298,"WAPE = Σ |预测 − CA| / Σ CA；最大误差 = max |预测 / CA − 1|。")
labels=["原生 A4W4","展开 A8W4","MXFP4 量化","显式 AntiMxQuant","显式 GroupedMatmulV5"]
indices=[6,7,1,3,4]
def err(row,col):
    nums=re.findall(r"([\d.]+)%",row[col])
    return [float(x) for x in nums]
a=axis(162,404,353,175)
for j,(chip,color) in enumerate([("950DT",teal),("950PR",blue)]):
    a.barh([i+.31*j for i in range(5)],[err(summary[i],3+j)[0] for i in indices],height=.27,color=color,label=chip)
a.set_yticks([i+.15 for i in range(5)],labels);a.invert_yaxis();a.set_xlabel("冻结独立 WAPE（%）");a.legend(loc="upper right",ncol=2,fontsize=8);a.grid(axis="x",alpha=.15)
note(359,"图 4   五条双端留出路线，每条在每颗芯片上各验证 10 个新形状。图中展示 WAPE；逐项最大误差见附表。")
para(646,"PR 的展开 A8W4 平均误差 11.32%，但 [8192,128,512] 为 13.312 / 8.497 µs，高估 56.67%。这一行值得单独核查 AIV 展开权重、UB→L1 交接、Cube 消费的重叠。整窗差额本身不能证明是哪一段出错。",11.8)
source("docs/error-analysis.md、docs/errors/matmul_mx_mixed.md；基础集 2,180 个可比窗，开发 2,060、冻结独立 120，后增 Flash 原生窗单列。")

page("07  /  归因实验","同一组候选页，读取顺序改变了首行","950DT，同一 Q64、1Mi 历史、每行 2048 个八键块、32 个计算团队。")
para(194,"三个输入使用同一发布版 MXFP4 原生 QSLI，打分、掩码和分核对齐。每核做两行，每行 32 次 MMAD。只把相同的页按行轮转，设备窗从 99.027 降到 72.210 µs。两组首行 MMAD 窗中位数相差 26.09 µs，第二行只差 0.12 µs。",11.7)
picture("figures/flash-qsli-q64-layout.png",276,324)
note(607,"图 5   上栏为估计、下栏为原生 CA。竖纹表示发起指令，黑色竖线表示全设备整窗结束。中图为终版盲测。")
para(657,"因此，候选重合度与各团队的起始 tile 共同影响首片。进入已预取的续片后，沿用旧 DT 观测的 1343.61 周期并检查目标资源下界。“半数页移动”保留至二次冻结后揭盲，预测 92.129 µs、CA 84.182 µs（+9.44%）。生产页序未知时默认按中点情景处理。",11.2)
source("results/v1.13/evaluation.json；calibration/candidate_indexer/v113_control.md。其余两端点参与终版机制选择。")

page("08  /  系统评估","33 组请求怎样比较","单卡 rank 0 捕获图接入原系统离散事件仿真。每组三种后端共享工作负载、拓扑和图依赖。")
para(197,"模型从每个节点取得并行切分后的单卡 shape。32Ki、B=8 的 prefill 共有 262,144 个输入 token；若每请求每块 4096，一次投影有 32,768 行，要运行八块。Decode 的单卡工作量另受 batch、推测校验行数和 KV 历史影响。",11.8)
a=axis(95,344,423,180)
models=[('v4-pro','DeepSeek-V4 Pro'),('v41-flash','V4.1 Flash'),('kimi-k31','Kimi K3.1 Pro')]
for k,(model,name) in enumerate(models):
    r=[x for x in requests if x['model']==model and x['batch']=='1']
    r.sort(key=lambda x:int(x['prompt']))
    a.plot(range(4),[float(x['request_v113_full_s']) for x in r],marker=['o','s','D'][k],color=[blue,teal,amber][k],label=name)
a.set_yscale("log");a.set_xticks(range(4),["32Ki","128Ki","512Ki","1Mi"]);a.set_ylabel("完整请求，秒（对数刻度）");a.grid(axis='y',alpha=.2);a.legend(fontsize=8)
note(285,"图 6   B=1、TP8 的系统预测，含 prefill 和 1024 个输出 token。1Mi 之后的 decode 超出预设位置，作为情景外推列入。")
para(588,"33 组中有 18 组比历史 Roofline 快，所有组都满足同精度、同算法的峰值下界。历史后端可能按完整 BF16 表扫描，而实际 Flash 消费层只在候选上做 MXFP4 点积。比较增益时必须说明换了哪些工作、哪些只换了定价。",11.5)
para(682,"相对 v1.12，只有 5 组长上下文 Flash 改变。1Mi/B1 的整请求从 93.079 s 降到 87.848 s。按 CA 续行节奏外推的末块为 458.052–458.655 ms；默认系统预测 458.229 ms。两种外推彼此接近，整段真实执行时间仍待观测。",11.4)
source("docs/context-sweep.md；calibration/candidate_indexer/v113_context_requests.csv。附表列出全部 33 组。")

page("09  /  优化空间","下一轮把验证成本花在哪里","按已经看到的误差、系统可获得的输入，以及一次试验能区分的问题排序。")
for y,title,body,action in [
 (196,"1   优先打通调用前序","mHC Post 的同一批 239 个窗，给出同流前序后，DT / PR 的 WAPE 从 27.94% / 28.02% 降至 8.38% / 6.83%。这是比增加 shape 常数更直接的改善。","接入处按卡、按流保留前驱和已驻留核；在首次、扩核、热态三组上回放，再冻结验证。前序不确定时保留仅 shape 的误差带。"),
 (344,"2   定位 PR 的 A8W4 长尾","[8192,128,512] 的预测比 CA 多 4.815 µs。低位 W 需要展开，搬运、展开和点积可以同时进行。","先从旧 trace 取每核首个 W 页交接、末次 MMAD 退休和写尾，查误差在哪段出现；冻结后选邻近 K 页边界和宽 N 对照。"),
 (492,"3   优先补能改变设计决策的硬件量","970DT 的 FA 示例对 Vector 吞吐敏感；同一配置的矩阵乘对 L2 更敏感。QSLI 的 1 AIC + 2 AIV 团队在 120 / 120 配置上最多组成 60 队。","对不同核配比、搬运端口和 UB 预算分别给情景；有新硬件时先测受影响的阶段。FA 与 QSLI 的配核规则应分别验证。"),
 (640,"4   用分核周期压缩系统采样","Kimi 三组 decode 已有 147 轮全量基准。按 64 个 AIV 的行数余数分组后，每组只需 16、8、4 个轮级样本，总时间偏差均低于 0.0023%。","保留整轮回放作参照。切换精度、分核或缓存分页时重新核对周期。Flash 若继续采集，优先补 PR 的同形换页序或完整调用边界。")]:
    heading(y,title);para(y+32,body,10.8,leading=16);para(y+85,action,10.3,color=muted,leading=15)
source("docs/errors/mhc_post-stream.md、docs/context-sweep.md；优化建议与已完成的测量分列。")

page("10  /  使用与边界","如何在系统仿真中使用结果","输入选择会改变模型的含义。同形状的不同执行精度、融合实现、张量步长及前序可能得到不同价格。")
heading(207,"最小调用")
rect(47,245,501,81,light)
for i,s in enumerate(['from operator_model import estimate','r = estimate("matmul", (1280, 2048, 512), dtype="bf16",','                     chip="Ascend950DT_9582", memory_policy="cold_io")']):
    txt(59,255+i*20,s,10)
para(345,"读取 r.latency_us、r.active_aic、r.active_aiv 和 r.work 的分精度工作量，再由系统排跨流并发与通信。在同一个进程复用芯片配置；对有序的 RMSNorm 和 mHC Post，通过 OperatorStream 提供前序。")
heading(449,"预测成本与校准成本")
para(484,"基础集中 4,360 次 CPU 查询，中位 0.173 ms、P95 为 3.058 ms、最大 18.910 ms。新 QSLI 发布包的 1,200 次查询最大 7.99 ms。一次新增 Q64 CA 约需 26 分钟（含导出），所以采集重点是可以检验不同解释的少量形状。")
heading(596,"覆盖范围")
para(631,"原 27 组 decode 的图核对了 80 个原始调用名，后续扩展矩阵覆盖 86 个。通信按互连场景定价，视图可为零设备时间，部分融合只取得组件窗。图上的全部节点能报价，与每一项都有原生设备窗，是两种不同的覆盖指标。",11.8)
note(720,"阅读附表：先找相同的算子、位宽和实现，再看数据角色。N/A 表示缺少可比较的价格或真值。")
source("README.md、docs/integration.md、docs/system-full-evidence.md、docs/error-analysis.md。")

page("11  /  本次落地优化","获得候选地址统计的成本可以降低","模型只需要一项候选重合度。若系统保留了页表，可以在请求采样时精确提取，然后复用于多个设备报价。")
para(200,"对同一请求的两行，比较相同序号的有效候选块。分子是映射到同一物理页、同一页内偏移的次数；分母是双方都有有效候选的序号数。行长不齐时，逐元素重复更新分母会增加主机开销。")
panel(308,"一段连续序号有 n 行有效，长度为 d；该段对分母贡献 d × n × (n−1) / 2。")
para(376,"改动后按行长变化分段求分母，分子继续逐条核对地址。公开接口、页表别名处理和返回值保持原语义。300 组随机不等长输入各用逻辑与物理页表口径复核，600 组完整结果与优化前相等。",11.7)
bench=get(DATA/'layout-prepare-benchmark.json')["rows"]
table(475,["64 行输入","原实现中位 ms","优化后中位 ms","耗时降低"],[
    [r["layout"],f'{r["baseline_median_ms"]:.2f}',f'{r["optimized_median_ms"]:.2f}',
     f'{100*(1-r["optimized_median_ms"]/r["baseline_median_ms"]):.1f}%'] for r in bench],[131,131,131,108])
para(607,"同序与错序均为每行 2048 项；不等长从 2048 开始逐行减 13 项。单进程交替执行，每种输入预热后取 10 次中位数。计时只含统计函数，数据是构造的 CPU 输入。设备定价仍使用 v1.13.1 的冻结公式，本次没有增加 CA 采集。",11.8)
source("calibration/candidate_indexer/layout-prepare-benchmark.json；src/operator_model/operators/attention/candidate_layout.py。")
for off in range(0,len(summary),12):
    subset=summary[off:off+12]
    page("附表 A  /  校准索引",f"各实现的双端误差  ·  {off//12+1}","每端列出“形状数/窗口数 · WAPE / 最大误差”。百分比只对有可比完整窗的实现计算。")
    vals=[]
    for name,dtype,role,dt,pr,*_ in subset:
        # Keep full precision names on a second line to preserve readable text.
        vals.append([name,dt,pr,dtype,role])
    y=213
    txt(48,y,"算子与执行合同",10,teal,True);txt(298,y,"950DT",10,teal,True);txt(425,y,"950PR",10,teal,True)
    for i,(name,dt,pr,dtype,role) in enumerate(vals):
        y=245+i*41
        if i%2==0:rect(47,y-4,501,40,light)
        fs=10.1
        while width(name,fs)>238:fs-=.3
        txt(50,y,name,fs)
        for x,v in [(298,dt),(425,pr)]:
            fs=9.4
            while width(v,fs)>118:fs-=.2
            txt(x,y,v,fs)
        text=dtype+"    /    "+role;fs=8.35
        while width(text,fs)>493:fs-=.2
        txt(50,y+19,text,fs,muted)
    source("docs/error-analysis.md 开头的逐合同总表。同一物理窗可用于检验不同输入合同，跨行不宜累加。")
for model,name in models:
    rs=[r for r in requests if r["model"]==model]
    page("附表 B  /  端到端系统预测",name,"每行一个配置。秒数包括设备 / 通信的离散事件调度，无法直接作为真机请求误差。")
    vals=[]
    for r in rs:
        length=int(r['prompt'])//1024
        dims=f"{length}Ki" if length<1024 else "1Mi*"
        vals.append([dims,r['batch'],r['tp']+'/'+r['ep'],*[f"{float(r[k]):,.2f}" for k in
             ('prefill_v113_full_s','decode_v113_full_s','request_v113_full_s','request_v113_peak_s','request_legacy_whole_s')], '是' if r['fits_144GB']=='True' else '否'])
    table(236,["输入","B","TP/EP","prefill","decode","请求","同精度峰值","历史后端","可放入¹"],vals,[52,34,53,60,56,60,68,66,52],step=31,size=9.5)
    note(630,"表中输出长均为 1024。请求 = prefill + decode，先用原始值求和再分别舍入。峰值和历史两列均为整请求时间。")
    note(681,"¹ 可放入依据原任务 144 GB 预算的静态估计。* 1Mi 输入之后继续生成超出原模型预设位置，decode 报价为外推。")
    source("calibration/candidate_indexer/v113_context_requests.csv；docs/context-sweep.md 提供完整配置、采样节点和口径。")
pdf.savefig(fig);plt.close(fig);pdf.close()
print(f"{OUT} ({count} pages)")
