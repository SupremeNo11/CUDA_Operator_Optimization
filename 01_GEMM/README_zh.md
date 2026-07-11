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