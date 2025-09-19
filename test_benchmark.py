import subprocess

# 依次要执行的命令（写在列表里）
commands = [
    "CXX=hipcc cmake -DBUILD_BENCHMARK=ON -DAMDGPU_TARGETS=gfx942 ../.",
    "make -j4",
    "make install",
    "./benchmark/benchmark_device_merge_sort"
]
workdir = "/xxcsinaccx__hipcub.git/project/hipcub/build"
for cmd in commands:
    print(f"running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=workdir)
    if result.returncode != 0:
        print(f"fail: {cmd}")
        break
