"""Preserve the reference host's inputs/oracle; allow the 3rd and later D tile."""
from pathlib import Path
import os
import subprocess

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'results/v0.23/hc-timing/raw'
OLD=ROOT/'build/hcpost23'
SOURCE=ROOT/'development/validation/hcpost23/host.cpp'

if __name__=='__main__':
    out=RAW/'build';out.mkdir(exist_ok=True)
    origin=SOURCE.read_text().replace('"../../../smoke/extension_indexer_mhc.cpp"',f'"{ROOT}/smoke/extension_indexer_mhc.cpp"')
    before='int t = dim(argv[1]), h = dim(argv[2]);'
    after='''auto checked_dimension = [](const char *arg) {
            size_t end = 0;
            long long n = std::stoll(arg, &end);
            if (end != std::strlen(arg) || n < 1 || n > 1048576)
                throw std::runtime_error("Invalid dimension");
            return static_cast<int>(n);
        };
        int t = checked_dimension(argv[1]), h = checked_dimension(argv[2]);'''
    assert origin.count(before)==1
    main=out/'host.cpp';main.write_text(origin.replace(before,after))
    kernel=out/'kernel'
    if not kernel.exists():kernel.symlink_to(OLD/'kernel',target_is_directory=True)
    cann=Path(os.environ['ASCEND_HOME_PATH'])
    subprocess.run(['g++','-std=c++17','-O2',str(main),'-I'+str(OLD/'host/autogen'),'-I'+str(cann/'include'),
                    '-L'+str(OLD/'vendors/hcpost/op_api/lib'),'-L'+str(cann/'lib64'),'-Wl,-rpath-link,'+str(cann/'lib64'),
                    '-lcust_opapi','-lopapi','-lnnopbase','-lascendcl','-ldl','-o',str(out/'host_probe')],check=True)
