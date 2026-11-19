"""Compile nested versus released TPipe lifetimes using the real 950 cube headers."""
import re,subprocess
from packages import HERE,ROOT,RAW,OUT,save,probe,vendor
def main():
    cann=probe.CANN;asc=cann/'aarch64-linux/asc';runner=['bash',str(ROOT/'scripts/cann-run.sh')]
    src=HERE/'pipe_control.cpp';sdk=asc/'impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h'
    catlass=vendor('release')/'op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/common'
    flags=['-I'+str(catlass),'-xcce','-std=c++17','-O2','-D__DAV_C310__','--cce-aicore-arch=dav-c310-cube','--cce-aicore-only',
            *['-I'+str(f) for f in [asc/'include',asc/'include/basic_api',asc/'include/highlevel_api',asc/'include/utils',asc,cann/'include']]]
    rows=[]
    for key,i in [('single',0),('nested',1),('destroy_before_second',2),('reuse_reset',3),('real_wheel_catlass_resource',4),('release_before_wheel_resource',5)]:
        obj=RAW/('pipe-'+key+'.o')
        subprocess.run(runner+['bisheng',*flags,'-DPIPE_EXPERIMENT='+str(i),'-c',str(src),'-o',str(obj)],check=True)
        out=subprocess.run(runner+['llvm-objdump','-d',str(obj)],text=True,capture_output=True,check=True).stdout
        hits=[dict(offset='0x'+a,word=w,opcode=n) for a,w,n in re.findall(r'^\s*([\da-f]+):\s+([\da-f]+)\s+((?:SET|WAIT)_FLAG\.M\.MTE1)\s',out,re.M)]
        count=next((i for i,r in enumerate(hits) if r['opcode'].startswith('WAIT')),len(hits))
        rows.append(dict(source_case=key,definition=i,instruction_offsets=hits,linear_set_opcodes_before_first_wait_opcode=count,
            compiler_elf_sha256=probe.digest(obj),readout='ELF text/disassembly order, may contain conditional branches. Initial SET prefix before first possible WAIT appears on the actual physical AIC case; other conditional wait instructions must not be counted as observed retirements.'))
    assert [r['linear_set_opcodes_before_first_wait_opcode'] for r in rows]==[3,6,3,3,6,3]
    save(OUT/'compiler-controls.json',dict(compiler=probe.rel((cann/'bin/bisheng').resolve()),compiler_sha256=probe.digest(cann/'bin/bisheng'),
         sdk_sha256=probe.digest(sdk),catlass_resource=probe.rel(catlass/'catlass/arch/resource.hpp'),catlass_resource_sha256=probe.digest(catlass/'catlass/arch/resource.hpp'),sdk_file=probe.rel(sdk),sdk_lines={'Ctor_Init':84,'Init_duplicate_guard':103,'Init_M_MTE1_triple':111,'Destroy_wait_triple':802,'Reset':821},
         source=probe.rel(src),source_sha256=probe.digest(src),compile_flags=flags,rows=rows,additional_ca=0,
         scope='Three alternatives to overlapping TPipe ownership and the defective lifetime illustrated in a standalone copy-independent SDK translation unit. No KDA code was changed and no new instruction cost is represented.'))
    print([(r['source_case'],r['linear_set_opcodes_before_first_wait_opcode']) for r in rows])
if __name__=='__main__':main()
