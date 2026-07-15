# GEMM 是什么？

---

通用矩阵乘法，标准形式：

$$
C = \alpha A×B + \beta C
$$

其中，$A=M×K$，$B=K×N$，$C=M×N$

在深度学习计算中，通常形式是：$C=A×B$，即：$\alpha = 1，\beta = 0$

# AI模型里面的GEMM

--- 

AI模型中含有大量的的GEMM，大约占了80%。
比如：
- TransFormer中单的自注意力机制，$QK^T$就是GEMM。
- 前馈FFN中，$WX_1$和$WX_2$也是GEMM。

因此，GEMM的计算速度直觉决定AI模型的推理速度。

# GEMM优化方法

---

主要有两类优化方法：

1) 基于算法分析的优化方法，在很早之前，就有人根据矩阵乘法计算特性，从数学角度优化，典型的算法包括`Strassen`算法和`Coppersmith-Winograd`算法。
2) 基于软件优化的方法：从计算机的线程层次和内存层次优化矩阵计算过程。

# CPU三重循环怎么写？

---

`C`的每个元素等于`A`的第`i`行与`B`的第`j`列的点积。

$$
C_{ij}=\sum_{k=0}^{K-1}A_{ik}·B_{kj}
$$

外层`i`行，中层`j`列，内层`k`求和。

## 一个具体的例子

设：
$$
A=\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix},
B=\begin{bmatrix}
7 & 8 \\
9 & 10 \\
11 & 12
\end{bmatrix}
$$

`A`行访问连续、`B`列访问步长为`N`，不连续

# GPU线程层次理解

---

### 核心层级架构 (The CUDA Hierarchy)

1. **Kernel Launch (核函数启动) = 1 个 Grid**
当用 `<<< >>>` 语法调用一个核函数时，GPU 就会在底层创建一个 Grid。这个 Grid 包含了本次计算任务所需要的所有线程。
2. **1 个 Grid = 多个 Block (线程块)**
Grid 是一个逻辑上的大容器，它被划分为多个 Block。也就是代码里的 `dim3 grid(...)` 参数。
3. **1 个 Block = 多个 Thread (线程)**
每个 Block 内部又包含了多个 Thread，这就是具体干活的工人。也就是代码里的 `dim3 block(...)` 参数。

---

### 代码与概念的直接映射

矩阵乘法启动代码：

```cpp
dim3 block(16, 16);
dim3 grid((W + 15) / 16, (H + 15) / 16);

// 这一行代码执行时，就诞生了【1个 Grid】
MatrixMulTiled<<<grid, block>>>(A_d, B_d, C_d, H, W, K); 

```

当这行代码执行时，GPU 会发生以下事情：

* **诞生**：GPU 接收到命令，为 `MatrixMulTiled` 这个核函数生成了 **1 个完整的 Grid**。
* **分配**：这个 Grid 的大小和维度，完全由 `<<<grid, ...>>>` 里的第一个参数决定。
* **消亡**：当这个 Grid 里面的所有 Block、所有 Thread 都把计算任务跑完（比如矩阵乘法的所有元素都算完写回全局内存），这个 Kernel 就算执行完毕，**这个 Grid 也就随之消亡**。

### 关键延伸：如果调用多次 Kernel 呢？

注意区分“核函数代码”和“核函数启动”。

如果在 `main` 函数里写了一个 `for` 循环，把同一个 Kernel 连续调用了 5 次：

```cpp
for(int i=0; i<5; i++){
    MatrixMulTiled<<<grid, block>>>(...);
}

```

那么，GPU 实际上是按顺序创建了 **5 个独立的 Grid**（每次启动对应一个 Grid）。它们是相互隔离的执行批次，哪怕它们用的是同一段逻辑代码。

