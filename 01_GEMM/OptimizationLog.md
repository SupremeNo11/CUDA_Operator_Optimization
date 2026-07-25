# 优化日志

记录每一次优化过程和性能分析。

---

## 第一次优化：GPU运行通用矩阵乘法
---
### 优化步骤
1. Host new一块内存初始化，Device cudaMalloc 一块同样大小的内存
2. cudaMemcpy Host Mem 拷贝到 Device Mem
3. GEMM 计算
4. cudaMemcpy Device Mem 拷贝到 Hsot Mem
5. cudaFree 释放Device Mem

### 性能分析（nvprof）
```bash
==23561== Profiling application: ./gemm_optimization
==23561== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   92.89%  85.088ms         1  85.088ms  85.088ms  85.088ms  gemm_kernel_comm(float*, float*, float*, int, int, int)
                    3.86%  3.5353ms         1  3.5353ms  3.5353ms  3.5353ms  [CUDA memcpy DtoH]
                    3.25%  2.9781ms         2  1.4891ms  1.4838ms  1.4943ms  [CUDA memcpy HtoD]
      API calls:   58.78%  150.84ms         1  150.84ms  150.84ms  150.84ms  cudaSetDevice
                   35.83%  91.931ms         3  30.644ms  1.5399ms  88.774ms  cudaMemcpy
                    4.77%  12.234ms         2  6.1170ms  4.6870us  12.229ms  cudaEventRecord
                    0.30%  781.25us         3  260.42us  70.566us  363.21us  cudaFree
                    0.14%  357.09us         3  119.03us  55.796us  199.82us  cudaMalloc
                    0.06%  153.30us         1  153.30us  153.30us  153.30us  cudaLaunchKernel
                    0.05%  126.46us       114  1.1090us     105ns  48.351us  cuDeviceGetAttribute
                    0.03%  89.424us         1  89.424us  89.424us  89.424us  cudaGetDeviceProperties
                    0.02%  42.643us         4  10.660us     308ns  34.182us  cudaDeviceGetAttribute
                    0.01%  17.106us         2  8.5530us     785ns  16.321us  cudaEventCreate
                    0.00%  10.493us         1  10.493us  10.493us  10.493us  cuDeviceGetName
                    0.00%  6.4770us         1  6.4770us  6.4770us  6.4770us  cuDeviceGetPCIBusId
                    0.00%  4.1080us         1  4.1080us  4.1080us  4.1080us  cuDeviceTotalMem
                    0.00%  3.9530us         1  3.9530us  3.9530us  3.9530us  cudaEventSynchronize
                    0.00%  1.3570us         3     452ns     221ns     836ns  cuDeviceGetCount
                    0.00%  1.1650us         1  1.1650us  1.1650us  1.1650us  cudaEventElapsedTime
                    0.00%     707ns         2     353ns     144ns     563ns  cuDeviceGet
                    0.00%     688ns         1     688ns     688ns     688ns  cudaGetDeviceCount
                    0.00%     520ns         1     520ns     520ns     520ns  cuModuleGetLoadingMode
                    0.00%     330ns         1     330ns     330ns     330ns  cudaDriverGetVersion
                    0.00%     192ns         1     192ns     192ns     192ns  cuDeviceGetUuid
                    0.00%     179ns         1     179ns     179ns     179ns  cudaRuntimeGetVersion
```

整份报告被明确地分成了两大区块：**GPU activities（GPU 端活动）** 和 **API calls（CPU 端调用的 CUDA API）**。

* **Time(%)**：该项操作在当前分类（GPU 或 CPU）中的总耗时占比。**这是寻找性能瓶颈（找茬）最关键的指标。**
* **Time**：该项操作的累计总耗时。
* **Calls**：该函数被调用的总次数。
* **Avg / Min / Max**：单次调用的平均、最短、最长耗时。

---

#### 核心分析：GPU activities (GPU 到底在忙什么？)

这部分记录的是 GPU 硬件真正干活的时间。 GPU 总共就做了三件事：

1. **`gemm_kernel_comm` (绝对核心瓶颈)**
* **占比与耗时**：高达 **92.89%**，单次运行耗时 **85.088 毫秒**。
* **解读**：这说明 GPU 绝大部分时间都在死磕矩阵乘法的数学计算。在初期的 Naive（朴素）版本中，计算耗时远大于数据搬运耗时，这是非常正常的现象。


2. **`[CUDA memcpy HtoD]` (数据传入)**
* **占比与耗时**：3.25%，总耗时近 3 毫秒，**Calls 为 2 次**。
* **解读**：HtoD 意思是 Host to Device（内存拷入显存）。这里的 2 次调用完美对应了程序中把 **矩阵 A** 和 **矩阵 B** 拷贝到 GPU 的操作。


3. **`[CUDA memcpy DtoH]` (结果传出)**
* **占比与耗时**：3.86%，耗时 3.5 毫秒，**Calls 为 1 次**。
* **解读**：DtoH 意思是 Device to Host（显存拷回内存）。这 1 次调用对应的是把算好的 **结果矩阵 C** 拷回 CPU 进行结果验证或打印。



---

#### 辅助分析：API calls (CPU 端的隐形开销)

这部分记录的是 CPU 执行 `cuda...` 代码时的耗时。

1. **`cudaSetDevice` 为什么耗时这么长？(占 58.78%, 150 毫秒)**
* **解读**：在任何 CUDA 程序中，CPU 第一次调用与 GPU 相关的 API 时，NVIDIA 驱动会在底层进行 **CUDA Context Initialization（上下文初始化）**，从 CUDA 12.0 开始，调用 cudaInitDevice() 或 cudaSetDevice() 会立刻、显式地初始化运行时和对应的 Context。这个“开机”过程非常耗时（通常在 100ms 到 500ms 之间）。
* **对策**：在做严谨的性能测试时，通常会在计时开始前，先随便跑一个 `cudaFree(0);` 或空的核函数来“预热” GPU，从而把这 150 毫秒的开销剥离出去。


2. **`cudaMemcpy` API 的异常耗时 (单次 Max 达到 88.7 毫秒)**
* **解读**： API 层面的 `cudaMemcpy` 总耗时是 91 毫秒，但上面 GPU 真正搬运数据的时间加起来才 6.5 毫秒。这中间巨大的时间差去哪了？
* **原因**：因为使用的是**可分页内存（Pageable Memory）**（即普通的 `new` 或 `malloc`）。当调用 `cudaMemcpy` 时，CUDA 驱动在底层必须先偷偷在主机端申请一块临时的“锁页内存（Pinned Memory）”，把数据搬过去，然后再让 GPU 执行真正的物理拷贝。那个高达 88.7 毫秒的 Max 时间，极大概率是系统在做这层隐形的内存转换。



#### 总结与优化方向

* **数据拷贝层面不需要操心**：虽然 API 调用有隐形开销，但物理拷贝时间仅占总时间的不到 7%，目前完全不需要搞 CUDA 流（Streams）这种高级异步操作。
* **全力优化 `gemm_kernel_comm**`：85 毫秒的纯计算时间是巨大的优化空间。接下来的目标，就是想办法修改内核代码，把这个数字降到 10 毫秒甚至更低级别！



---

## 第二次优化：使用共享内存降低带宽延迟
---
### 优化步骤
1. 对目标矩阵按照Block大小进行分块，每个Block负责计算一部分
2. 轮询拷贝，全局内存拷贝到共享内存当中
3. 线程同步
4. 轮询计算
5. 线程同步
6. 写回Device内存

### 性能分析（nvprof）

```bash
Program completed.
==494774== Profiling application: ./gemm_optimization
==494774== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.52%  86.842ms         1  86.842ms  86.842ms  86.842ms  gemm_kernel_comm(float*, float*, float*, int, int, int)
                   28.10%  39.671ms         1  39.671ms  39.671ms  39.671ms  void gemm_kernel_smem<int=16>(float*, float*, float*, int, int)
                    8.20%  11.575ms         2  5.7877ms  3.4251ms  8.1504ms  [CUDA memcpy DtoH]
                    2.18%  3.0756ms         2  1.5378ms  1.5109ms  1.5647ms  [CUDA memcpy HtoD]
      API calls:   52.82%  126.52ms         2  63.259ms  39.673ms  86.845ms  cudaEventSynchronize
                   39.84%  95.428ms         1  95.428ms  95.428ms  95.428ms  cudaSetDevice
                    6.67%  15.981ms         4  3.9953ms  1.6331ms  9.0730ms  cudaMemcpy
                    0.31%  746.30us         3  248.77us  88.499us  334.27us  cudaFree
                    0.13%  312.13us         3  104.04us  53.709us  179.82us  cudaMalloc
                    0.07%  158.03us         2  79.013us  14.092us  143.94us  cudaLaunchKernel
                    0.07%  157.15us       114  1.3780us     108ns  84.502us  cuDeviceGetAttribute
                    0.04%  88.226us         1  88.226us  88.226us  88.226us  cudaGetDeviceProperties
                    0.02%  49.879us         4  12.469us     866ns  37.432us  cudaDeviceGetAttribute
                    0.01%  21.762us         4  5.4400us  1.8380us  14.016us  cudaEventRecord
                    0.01%  19.202us         1  19.202us  19.202us  19.202us  cuDeviceTotalMem
                    0.01%  18.205us         2  9.1020us  1.2250us  16.980us  cudaEventCreate
                    0.01%  14.753us         1  14.753us  14.753us  14.753us  cuDeviceGetName
                    0.00%  4.8770us         1  4.8770us  4.8770us  4.8770us  cuDeviceGetPCIBusId
                    0.00%  4.4410us         2  2.2200us  1.6830us  2.7580us  cudaEventElapsedTime
                    0.00%  1.7230us         3     574ns     195ns  1.0620us  cuDeviceGetCount
                    0.00%     826ns         2     413ns     344ns     482ns  cuDeviceGet
                    0.00%     674ns         1     674ns     674ns     674ns  cudaDriverGetVersion
                    0.00%     553ns         1     553ns     553ns     553ns  cudaGetDeviceCount
                    0.00%     338ns         1     338ns     338ns     338ns  cudaRuntimeGetVersion
                    0.00%     291ns         1     291ns     291ns     291ns  cuModuleGetLoadingMode
                    0.00%     222ns         1     222ns     222ns     222ns  cuDeviceGetUuid
```

优化后的kernel计算耗时39.671ms，优化前耗时86.842ms，速度快了将近1倍多。