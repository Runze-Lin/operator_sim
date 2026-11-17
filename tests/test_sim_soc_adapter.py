"""Exercise the real interposition ABI without a CANN runtime or CA run."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('g++'), 'g++ required for native adapter test')
class AdapterTests(unittest.TestCase):
    def test_gating_errors_and_buffer_bounds(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'runtime.cpp').write_text(r'''
#include <cstdint>
#include <cstdlib>
#include <cstring>
extern "C" int32_t rtGetSocVersion(char* b,uint32_t n) {
  if(std::getenv("FAKE_ERROR"))return 77;
  if(!b || n<3)return 88;
  std::strcpy(b,"PR"); return 0;
}
extern "C" int32_t halGetSocVersion(uint32_t d,char* b,uint32_t n) {
  if(d!=4)return 99;
  return rtGetSocVersion(b,n);
}
''')
            (p/'main.cpp').write_text(r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
extern "C" int32_t rtGetSocVersion(char*,uint32_t);
extern "C" int32_t halGetSocVersion(uint32_t,char*,uint32_t);
int main(int argc,char** argv) {
  char b[64]="untouched";
  int status=argc==2?rtGetSocVersion(b,std::atoi(argv[1])):halGetSocVersion(4,b,64);
  std::printf("%d:%s",status,b);
}
''')
            def compile(*args):
                subprocess.run(['g++',*map(str,args)],check=True,capture_output=True)
            compile('-shared','-fPIC',p/'runtime.cpp','-o',p/'runtime.so')
            compile('-std=c++17','-shared','-fPIC',root/'capture/sim_soc_adapter.cpp','-ldl','-o',p/'adapter.so')
            compile(p/'main.cpp',p/'runtime.so','-o',p/'test')
            env={k:v for k,v in os.environ.items() if k not in ('LD_PRELOAD','soc_version','OPERATOR_SIM_TARGET_SOC','FAKE_ERROR')}
            env['LD_PRELOAD']=str(p/'adapter.so')
            def run(size=None,**extra):
                return subprocess.run([str(p/'test')]+([] if size is None else [str(size)]),env={**env,**extra},check=True,capture_output=True,text=True).stdout
            self.assertEqual(run(64),'0:PR')
            target={'soc_version':'Ascend950DT','OPERATOR_SIM_TARGET_SOC':'Ascend950DT_9582'}
            self.assertEqual(run(64,**target),'0:Ascend950DT_9582')
            self.assertEqual(run(**target),'0:Ascend950DT_9582')
            self.assertEqual(run(64,**{**target,'soc_version':'Ascend950PR'}),'0:PR')
            self.assertEqual(run(64,**{**target,'OPERATOR_SIM_TARGET_SOC':'Ascend910B'}),'0:PR')
            self.assertEqual(run(64,**target,FAKE_ERROR='1'),'77:untouched')
            self.assertEqual(run(1,**target),'88:untouched')
            self.assertEqual(run(3,**target),'1:PR')
