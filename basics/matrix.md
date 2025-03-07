**矩阵**是高等代数学中的常见工具，也常见于统计分析等应用数学学科中。在人工智能的项目中，无论是使用机器学习，还是做数值优化，都会用到矩阵的知识。因此借助这篇文章，让我们来一起了解一下有关矩阵的一些基础概念。

# 标量，向量

 -   **标量**是一个**单独的数值**，通常用于表示一个量的大小。例如：
    
	  -   温度：25°C
	  -   质量：70kg
	  -   学生的考试分数：95分
		-   **数学表示**：    
		    -   用小写字母表示，如：a,b,xa, b, xa,b,x
		    -   例子：a=5a = 5a=5
		-   **特性**：    
		    -   只有**大小**，没有方向。
		    -   在计算机中可以用**单个数值（如整数、浮点数）**表示。
-   **向量**（vector）：向量是一个**有方向的数值集合**，可以表示多个数值组成的实体。 例如：    
    -   位移（不仅有大小，还有方向）
    -   速度（速度大小 + 方向）
    -   一个学生在三门考试中的分数 [85,90,78][85, 90, 78][85,90,78]
	-   **数学表示**：    
	    -   用小写粗体字母或带箭头的符号表示，如：**v\mathbf{v}v** 或 		**v⃗\vec{v}v**。
		    -   例子：
        -   **二维向量**（平面上的点）： v=[3，4]
        -   **三维向量**（空间中的点）： v=[3, 4, 5]
	-   **特性**：    
		   -   既有**大小**，又有**方向**。
		   -   在计算机中通常用**数组（list, NumPy array）**表示，如 `[3, 4]` 或 `[1, 2, 3]`。

- **在机器学习中的应用**
	-   **标量**：通常用来表示一个单独的值，例如 **学习率（0.01）** 或 **损失值（2.5）**。
	-   **向量**：
	    -   **输入特征**（特征向量）：一个样本的数据，如 `[身高, 体重, 年龄]`
	    -   **权重向量**：神经网络或机器学习模型的参数
	    -   **词向量（Word Embedding）**：NLP 中用向量表示单词的意义，如 `["king"] → [0.2, 0.8, 0.5, ...]`
	    - 
# **矩阵**（matrix）

矩阵是一个**由数值排列成行和列的二维数组**，可以用来表示数据、方程组、图像、机器学习模型的参数等。
## **1. 矩阵的定义**
将一些元素排列成若干行，每行放上相同数量的元素，就是一个矩阵（Matrix）。
数学上，一个 m×n 的矩阵是一个由m行和n列元素排列成的矩形阵列。
矩阵里的元素可以是数字、符号或数学式。

![enter image description here](https://github.com/xiaohuidu/AI/blob/master/images/Matrix.svg)

-   **数学表示**：  
    矩阵通常用大写字母表示，如 **A,B,C**。  
    例如，下面是一个 2×3（2 行 3 列）的矩阵：
    
    ![enter image description here](https://github.com/xiaohuidu/AI/blob/master/images/matrix1.png)
   

-   **表示方式**：
    
    -   行（Row）：矩阵的横向元素。
    -   列（Column）：矩阵的纵向元素。
    -   元素（Element）：矩阵中的每个数值，通常表示为 ![enter image description here](https://github.com/xiaohuidu/AI/blob/master/images/aij.png)，其中 i 是行号，j 是列号。
    
    例如，![enter image description here](https://github.com/xiaohuidu/AI/blob/master/images/a23.png)，表示第 2 行第 3 列的值是 6。
    

----------

## **2. 矩阵的维度**

矩阵的维度（大小）通常表示为 **m*n**，其中：

-   **m** = 行数
-   **n** = 列数

常见矩阵类型：
-   **行向量（Row Vector）**：只有 1 行，形如 **1×n1 \times n1×n**（如 [2,3,4][2, 3, 4][2,3,4]）
-   **列向量（Column Vector）**：只有 1 列，形如 **m×1m \times 1m×1**（如 [234]\begin{bmatrix} 2 \\ 3 \\ 4 \end{bmatrix}​234​​）
-   **方阵（Square Matrix）**：行数等于列数，如 **3×33 \times 33×3** 矩阵。

----------

## **3. 矩阵的运算**

矩阵运算在机器学习和 AI 计算中非常重要：

1.  **矩阵加法和减法**（要求矩阵大小相同）：
    
    A+B=[1234]+[5678]=[681012]A + B = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}A+B=[13​24​]+[57​68​]=[610​812​]
2.  **标量乘法**（矩阵中的每个元素乘以一个数）：
    
    2×A=2×[1234]=[2468]2 \times A = 2 \times \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}2×A=2×[13​24​]=[26​48​]
3.  **矩阵乘法**（行乘列，需要 Am×nA_{m \times n}Am×n​ 的列数等于 Bn×pB_{n \times p}Bn×p​ 的行数）：
    
    [1234]×[5678]=[1×5+2×71×6+2×83×5+4×73×6+4×8]=[19224350]\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \times 5 + 2 \times 7 & 1 \times 6 + 2 \times 8 \\ 3 \times 5 + 4 \times 7 & 3 \times 6 + 4 \times 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}[13​24​]×[57​68​]=[1×5+2×73×5+4×7​1×6+2×83×6+4×8​]=[1943​2250​]
4.  **矩阵转置（Transpose）**（行变列，列变行）：
    
    AT=[1234]T=[1324]A^T = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}AT=[13​24​]T=[12​34​]

----------

## **4. 矩阵在 AI 和机器学习中的应用**

矩阵在 AI 和 ML 中无处不在，例如：

-   **数据存储**：数据集通常表示为矩阵，每一行是一个样本，每一列是一个特征（如身高、体重）。
-   **图像处理**：图像是一个像素值矩阵（灰度图是 2D，彩色图是 3D）。
-   **神经网络**：权重和输入都是矩阵，神经网络训练时涉及大量矩阵运算。
-   **线性回归**：可以用矩阵求解模型的参数： Y=XW+bY = XW + bY=XW+b
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzQ0MTgzNDk5LC01NTQ3OTg4NDgsLTExMz
Y2OTcyODYsMjQ0NTQzNzc3LC05NjgzMjE0MjUsLTI1NjA2NTc3
OF19
-->