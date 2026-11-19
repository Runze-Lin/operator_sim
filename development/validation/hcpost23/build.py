"""Build just the pinned vLLM BF16/FP32 HcPost into a project-local CANN vendor."""
from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "runs/workloads19/sources/full/csrc"
WORK = ROOT / "build/hcpost23"
HOST = WORK / "host"
KERNEL = WORK / "kernel"
VENDOR = WORK / "vendors/hcpost"
TBE = VENDOR / "op_impl/ai_core/tbe"

def run(args, **kwargs):
    subprocess.run(args, check=True, **kwargs)

def stage(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)

def main():
    os.environ["TEST_DATA_ROOT_PATH"] = str(WORK/"tvm")
    build_python = ROOT / "runs/workloads19/sources/build-python"
    if build_python.is_dir():
        os.environ["PYTHONPATH"] = str(build_python) + os.pathsep + os.environ["PYTHONPATH"]
    WORK.mkdir(exist_ok=True,parents=True)
    run(["cmake","-Wno-dev","-G","Ninja","-S",SRC,"-B",HOST,"-DASCEND_COMPUTE_UNIT=ascend950","-DASCEND_OP_NAME=hc_post",
         "-DBUILD_OPEN_PROJECT=ON","-DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON","-DENABLE_BUILD_PKG=ON",
         "-DENABLE_OPS_HOST=ON","-DENABLE_OPS_KERNEL=ON","-DCMAKE_BUILD_TYPE=Release",
         "-DCUSTOM_ASCEND_CANN_PACKAGE_PATH="+os.environ["ASCEND_HOME_PATH"],"-DCANN_3RD_LIB_PATH="+str(SRC/"third_party"),
         "-DCMAKE_INSTALL_PREFIX="+str(WORK/"output")])
    run(["ninja","-C",HOST,"-j4","libcust_opapi.so","libcust_opsproto_rt2.0.so","generate_ops_info_ascend950",
         "hc_post_ascend950_src_copy","hc_post_ascend950_py_copy","hc_post_ascend950_mkdir"])
    # Upstream's tiling stub output uses the source checkout's build path even
    # with an out-of-tree build. Relocate this output and link input together.
    recipes = subprocess.check_output(["ninja","-C",HOST,"-t","commands","libcust_opmaster_rt2.0.so"],text=True)
    for recipe in recipes.splitlines():
        recipe = recipe.replace(str(SRC/"build/liboptiling.so"), str(HOST/"liboptiling.so"))
        if recipe.startswith("/usr/bin/c++"):
            args=shlex.split(recipe)
            obj=HOST/args[args.index("-o")+1]
            if not obj.exists():
                obj.parent.mkdir(exist_ok=True,parents=True)
                run(args,cwd=HOST)
        elif recipe.startswith(": && /usr/bin/c++") and any(x in recipe for x in ["-o libcust_opmaster_rt2.0.so","-o "+str(HOST/"liboptiling.so")]):
            args=shlex.split(recipe.split(" && ")[1])
            run(args,cwd=HOST)
    for src, dest in [(HOST/"libcust_opapi.so", VENDOR/"op_api/lib/libcust_opapi.so"),
                      (HOST/"libcust_opsproto_rt2.0.so",VENDOR/"op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"),
                      (HOST/"libcust_opmaster_rt2.0.so",TBE/"op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"),
                      (HOST/"autogen/aic-ascend950-ops-info.json",TBE/"config/ascend950/aic-ascend950-ops-info.json")]:
        stage(src,dest)
    alias = TBE/"op_tiling/liboptiling.so"
    if not alias.exists():
        alias.symlink_to("lib/linux/aarch64/libcust_opmaster_rt2.0.so")
    os.environ.update(ASCEND_CUSTOM_OPP_PATH=str(VENDOR),HI_PYTHON="python3",TILINGKEY_PAR_COMPILE="1",BIN_FILENAME_HASHED="1")
    binary=HOST/"binary/ascend950"
    params, = [p for p in (binary/"gen").glob("*_param.json") if "relocatable" not in p.name and
               [i["dtype"] for i in json.loads(p.read_text())["op_list"][0]["inputs"]]==["bfloat16","bfloat16","float32","float32"]]
    script, = [p for p in (binary/"gen").glob("*.sh") if params.name in p.read_text()]
    KERNEL.mkdir(exist_ok=True)
    run(["bash",script,binary/"src/hc_post/HcPost.py",KERNEL],cwd=WORK)
    bin_dir = WORK/"ascend950/bin"
    for item in KERNEL.glob("HcPost_*"):
        if item.suffix in (".json",".o"):
            stage(item,bin_dir/"hc_post"/item.name)
            stage(item,TBE/"kernel/ascend950/hc_post"/item.name)
    run(["python3",SRC/"cmake/scripts/util/ascendc_ops_config.py","-p",bin_dir,"-s","ascend950"])
    for item in bin_dir.glob("*.json"):
        stage(item,TBE/"kernel/config/ascend950"/item.name)
    cann=Path(os.environ["ASCEND_HOME_PATH"])
    run(["g++","-std=c++17","-O2",str(Path(__file__).with_name("host.cpp")),"-I"+str(HOST/"autogen"),"-I"+str(cann/"include"),
         "-L"+str(VENDOR/"op_api/lib"),"-L"+str(cann/"lib64"),"-Wl,-rpath-link,"+str(cann/"lib64"),
         "-lcust_opapi","-lopapi","-lnnopbase","-lascendcl","-ldl","-o",WORK/"host_probe"])

if __name__=="__main__":
    main()
