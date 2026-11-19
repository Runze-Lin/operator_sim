"""Compare the official tagged KDA source, installed source and failing binary. No CA reruns."""
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / 'runs/chunk-recovery/upstream'
CANN = ROOT / 'toolchains/cann/9.2.1/cann-9.2.1'
SRC = CANN / 'opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc'
PREFIX = 'attention/chunk_kda_fwd/'
TAGS = ('v9.2.1', 'v9.3.0-beta.1')
TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][\w$]*|\d+(?:\.\d+)?[\w]*|::|->|<<|>>|&&|\|\||==|!=|<=|>=|\+\+|--|\+=|-=|\*=|/=|##|\S', re.S)


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])


def tracked(tag):
    listing = git('ls-tree', '-r', tag, '--', PREFIX+'op_host', PREFIX+'op_kernel').decode().splitlines()
    return {name: fields.split()[2] for fields, name in (line.split('\t') for line in listing) if name.endswith(('.cpp', '.h', '.hpp'))}


def words(content):
    # Preserve C preprocessor boundary (an ordinary newline has no meaning in C++).
    content = re.sub(r'\\\r?\n', '', content.decode())
    code = '\n'.join(line+' __END_DIRECTIVE__' if line.lstrip().startswith('#') else line for line in content.splitlines())
    return [t for t in TOKEN.findall(code) if not t.startswith(('//','/*'))]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def host_file(path):
    return str(path.relative_to(ROOT))


def main():
    old, new = (tracked(tag) for tag in TAGS)
    if old.keys() != new.keys():
        raise ValueError('source file set changed')
    changed, installed = {}, {}
    for name in sorted(old):
        a = git('cat-file', '-p', old[name]); b = git('cat-file','-p',new[name]); path = name.removeprefix(PREFIX)
        local = SRC / 'chunk_kda_fwd' / path
        if local.exists():
            installed[path] = a == local.read_bytes()
        left, right = words(a), words(b)
        if left != right:
            ops = SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
            changed[path] = [{'before':' '.join(left[i:j])[:220], 'after':' '.join(right[k:l])[:220]} for tag,i,j,k,l in ops if tag != 'equal']
    audit = json.loads((ROOT/'results/v0.21/chunk/pr-flag-audit.json').read_text())
    dump = next(ROOT/file for file in audit['files'] if file.endswith('.dump'))
    elf = dump.with_name('aicore_binary.o')
    first_fault = min(pair['flagged']['cycle'] for pair in audit['pairs'])
    raw = [s for s in dump.read_text().splitlines() if (m:=re.match(r'^\[info\] \[(\d+)\]',s)) and int(m[1])<first_fault]
    if not raw:
        raise ValueError('unrecognized trace')
    range_evidence={}
    for name, lo, hi in [('outer',0x97018,0x97048),('second',0x99650,0x996b4),('later_static_waits',0x99b04,0x99b10)]:
        dis = subprocess.check_output(['bash',str(ROOT/'scripts/cann-run.sh'),'llvm-objdump','-d','--start-address='+hex(lo),'--stop-address='+hex(hi),str(elf)],text=True)
        range_evidence[name]=[m[1]+': '+m[2] for line in dis.splitlines() if (m:=re.match(r'^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(SET_FLAG\.\S+|WAIT_FLAG\.\S+|SFF0\.\S+|SBITSET1\.\S+)',line))]
    header = CANN/'aarch64-linux/asc/impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h'
    catlass = SRC/'common/catlass/include/catlass/arch/resource.hpp'
    devkit = ROOT/'runs/chunk-recovery/asc-devkit'
    new_header = devkit / 'impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h'
    pins = {tag: re.search(rb'set\(CATLASS_TAG_ID ([a-f0-9]+)\)', git('show', tag+':cmake/third_party/catlass.cmake'))[1].decode() for tag in TAGS}
    extra_refs = {ref: {'commit':git('rev-parse',ref+'^{commit}').decode().strip(), 'tree':git('rev-parse',ref+':'+PREFIX[:-1]).decode().strip()} for ref in ('v9.2.0', 'upstream/master')}
    info = dict(other_official_revisions=extra_refs, source_repo='https://gitcode.com/cann/ops-transformer', commits={tag:git('rev-parse',tag+'^{commit}').decode().strip() for tag in TAGS},
        trees={tag:git('rev-parse',tag+':'+PREFIX[:-1]).decode().strip() for tag in TAGS}, cpp_host_kernel_files=len(old), raw_blob_changes=sum(old[n]!=new[n] for n in old), cpp_token_changes=changed,
        catlass_default_git_pins=pins, asc_devkit_beta_commit=subprocess.check_output(['git','-C',str(devkit),'rev-parse','HEAD'],text=True).strip(), installed_tpipe_equals_beta_header=header.read_bytes()==new_header.read_bytes(), installed_source_compared=len(installed), installed_source_byte_mismatches=[p for p,ok in installed.items() if not ok],
        installed_support_headers_sha256={host_file(p):digest(p.read_bytes()) for p in (header,catlass)},
        trace_source=host_file(dump), lines_before_first_fault=len(raw), mmads_before_first_fault=sum(bool(re.search(r'\bMMAD(?:\.\w+)?\s+\{',s)) for s in raw),
        before_fault_instruction_sample=raw[-1], two_allocator_signatures=range_evidence,
        note='Lexical comparison strips comments, splices continued lines, preserves operators, string literals and preprocessor boundaries. It does not compare toolkit, compiler or simulator binaries.')
    (ROOT/'results/chunk-recovery/source-audit.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    main()
