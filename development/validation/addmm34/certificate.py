"""Native aclnnAddmm identity, biased numerical check and full-device timing certificate."""
import hashlib
import json
import mmap
import re
import struct
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT / 'development/validation/matmul33/replay.cc'
PKG = (ROOT / 'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel'
       '/ascend950/ops_nn/mat_mul_v3/MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance.o')
FAMILY = PKG.stem
CORE = re.compile(r'^(core\d+\.(?:cube|vec)core\d+)\s+([.\d]+)\s+[.\d]+\s*$', re.M)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _sha(stream)


def _sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(2**20), b''):
        h.update(block)
    return h.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def no_bias_rejections(shape):
    """Count how many outputs a correctly rounded *bias-free* GEMM would fail in the host oracle."""
    m, n, k = shape
    def bf16(value):
        x, = struct.unpack('I', struct.pack('f', value))
        return struct.unpack('f', struct.pack('I', ((x+32767+((x>>16)&1))>>16)<<16))[0]
    def val(i):
        return ((i % 17)-8)/16
    count = 0
    for i in range(min(17, m)):
        for j in range(min(17, n)):
            dot = sum(((k-1-q)//17+1)*val(i*k+q)*val(q*n+j+2) for q in range(min(17, k)))
            target = dot + val(j+5)
            if abs(bf16(dot)-target) > (1+abs(target))/64:
                count += ((m-1-i)//17+1)*((n-1-j)//17+1)
    return count


def mechanisms(path):
    hist = Counter(); copies = Counter(); fix = []; b_reads = []; btmoves = 0
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        for op in ('MMAD','MOV_L1_TO_BT','FIX_L0C_TO_DST','MOV_OUT_TO_L1_MULTI_ND2NZ'):
            key=f'"name":"{op}","ph":"X"'.encode(); pos = 0
            while (hit := buf.find(key, pos)) >= 0:
                begin=buf.rfind(b'{"args":', max(0, hit-8000), hit); end=buf.find(b'}',hit)
                require(begin >= 0 and end>hit, 'unknown instruction layout')
                detail=json.loads(json.loads(buf[begin:end+1])['args']['detail']); pos=end+1
                if any(detail.get(x) for x in ('is_sync','is_wait','is_duration_excluded')):
                    continue
                if op=='MMAD': hist[detail['m'],detail['n'],detail['k']]+=1
                elif op=='MOV_L1_TO_BT': btmoves+=1; copies['bias_bytes']+=detail.get('processed_bytes',0)
                elif op=='FIX_L0C_TO_DST': fix.append([detail['dst_addr'],detail['m'],detail['n']])
                else: b_reads.append([detail['src_addr'],detail['n'],detail['d'],detail['processed_bytes']])
    return hist, btmoves, copies['bias_bytes'], fix, b_reads


def certify(raw, sku, shape, cost):
    m,n,k=shape; log=(raw/'run.log').read_text(errors='replace')
    require(cost['exit_code']==0 and not cost['stop_reason'], 'capture failed or exceeded budget')
    expected=f'case=addmm api=aclnnAddmm M={m} N={n} K={k} self=bias dtype=BF16 alpha=1 beta=1 cubeMathType=0 workspace='
    require(log.count(expected)==1 and re.findall(r'^runtime_soc=(\S+)',log,re.M)==[sku], 'wrong input or SKU')
    require(all(s in log for s in ('SMOKE_PASS','All task success.','Profiling data parse finished.',
                'hbm3_8000M.ini' if '950DT' in sku else 'lpddr5_6400M_bailu.ini')), 'missing terminal or memory identity')
    require('SMOKE_FAIL' not in log and not re.search(r'\[ERROR\].*(execute_(set|wait)_flag|deadlock|invalid instruction)',log), 'kernel error')
    require(re.findall(r'^correctness count=(\d+) bad=(\d+) ',log,re.M)==[(str(m*n),'0')], 'incomplete output golden')
    rejected=no_bias_rejections(shape); require(rejected>0, 'bias not observable under host tolerance')
    names=re.findall(r'Start profiling on kernel: (\S+)',log); require(len(names)==1 and names[0].startswith(FAMILY+'_'), 'wrong kernel launch')
    table=CORE.findall(log); require(table and len(table)==len(dict(table)), 'incomplete official table')
    parts=list(raw.glob('profile/**/dump/aicore_binary.o')); require(len(parts)==1 and sha(parts[0])==sha(PKG), 'package ELF mismatch')
    base=parts[0].parent.parent/'simulator'; paths=sorted(base.glob('core*/trace.json'))
    require({p.parent.name for p in paths}==set(dict(table)), 'active roster mismatch')
    stars=list(raw.glob('stars_log*_task.dump')); require(len(stars)==1, 'missing single task file')
    pairs=re.findall(r'\[(\d+)\]\s+\[(TASK_BEGIN|TASK_DONE)\].*?stream_id:(\d+), task_id:(\d+)',stars[0].read_text(errors='replace'))
    require(len(pairs)==2 and [x[1] for x in pairs]==['TASK_BEGIN','TASK_DONE'] and pairs[0][2:]==pairs[1][2:] and int(pairs[0][0])<int(pairs[1][0]), 'incomplete task boundary')
    with tempfile.TemporaryDirectory() as tmp:
        exe=Path(tmp)/'reduce'; subprocess.run(['c++','-O3','-std=c++17',str(NATIVE),'-lcrypto','-o',str(exe)],check=True)
        lines=subprocess.check_output([str(exe),*map(str,paths)],text=True).splitlines()
    require(len(lines)==len(paths), 'missing shard result')
    cores={}; hist=Counter(); pieces=[]; b_addresses=Counter(); bias_bytes=0
    for line,path in zip(lines,paths):
        first,last,end,count,mma,fprint=line.split('\t'); first,last,end=map(float,(first,last,end)); name=path.parent.name
        require(abs(last-first-float(dict(table)[name]))<=.011, 'official lane duration mismatch')
        mh,bt,bbytes,fix,loads=mechanisms(path); hist.update(mh); pieces.extend(fix);bias_bytes+=bbytes
        require(sum(mh.values())==int(mma), 'MMAD count mismatch')
        rhs=[x[0] for x in loads if x[1:]==[k,n,2*k*n]]; b_addresses.update(rhs)
        cores[name]=dict(first_us=first,retire_us=last,end_us=end,events=int(count),mmad=int(mma),
                         bt_moves=bt,fixp_outputs=len(fix),full_rhs_reads=len(rhs),sha256=fprint)
    ca=max(x['retire_us'] for x in cores.values())-min(x['first_us'] for x in cores.values())
    tile_ops=sum(2*dm*dn*dk*ct for (dm,dn,dk),ct in hist.items())
    return dict(qualified=True,ca_us=ca,chip=sku,shape=shape,kernel=names[0],elf_sha256=sha(PKG),
        api='aclnnAddmm',self_shape=[n],output_checked=m*n,bad=0,outputs_rejecting_bias_free_bf16_dot=rejected,
        active_aic=sum('cubecore' in c for c in cores),active_aiv=sum('veccore' in c for c in cores),
        mmads=sum(hist.values()),mmad_geometry=[[*x,v] for x,v in sorted(hist.items())],
        issued_cube_ops=tile_ops,logical_cube_ops=2*m*n*k,bias_bt_moves=sum(x['bt_moves'] for x in cores.values()),
        bias_bt_processed_bytes=bias_bytes or None,full_rhs_descriptor_address_counts=dict(b_addresses),fixp_descriptors=len(pieces),
        trace=str((base/'trace.json').relative_to(ROOT)),trace_sha256=sha(base/'trace.json'),
        source_log_sha256=sha(raw/'run.log'),stars_sha256=sha(stars[0]),all_core_timing=cores,
        timing_scope='Start of earliest X to retirement of latest X on every official active core, exactly one physical END per core and one STARS task. Host and wrapper excluded.')
