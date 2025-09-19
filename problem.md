I use `projects/hipcub/benchmark/benchmark_device_merge_sort.cpp` to test `cub::DeviceMergeSort::SortKeys` performance. But the performance is too bad. How can I optimize the performance of this kernel. 
1. You can modify all files related to `cub::DeviceMergeSort::SortKeys`.
2. The code in `projects/hipcub/benchmark` is not allowed to be edited.
3. you can use the follow commands to build and run benchmark in Docker:     
# running command
You can run `python test_benchmark.py` to test the `cub::DeviceMergeSort::SortKeys` . Before run script test_benchmark.py, you need to read and check the code. The parameter `workdir` may need change.