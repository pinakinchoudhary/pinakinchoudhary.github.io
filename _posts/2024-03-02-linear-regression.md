---
title: Linear Regression
date: 2024-03-02 00:00:00 +0530
math: true
toc: true
categories: [Machine Learning, Linear Regression]
tags: [machine learning, linear regression, supervised learning]
---
## Notation

|General Notation| Description| Python (if applicable) |
| --- | --- | --- |
| $\mathbf{x}$ | Training Feature vector | `x_train` |
| $\mathbf{y}$ | Training targets values | `y_train` |
| $x^{(i)}$, $y^{(i)}$ | $i_{th}$ Training Example | `x_i`, `y_i` |
| $\hat{y}^{(i)}$ | Predicted value for $x^{(i)}$ | `y_hat_i` |
| $n$ | Number of training examples | `m` |
| $w$ | Model Parameter: weight | `w` |
| $b$ | Model Parameter: bias | `b` |
| $f_{w,b}(x^{(i)})$ | $f_{w,b}(x^{(i)}) = wx^{(i)}+b$ | `f_wb` |
| $\mathcal{J}(w,b)$ | Cost/Loss function evaluated at $w,b$ | `J_wb` |

Linear Regression is at its core, curve fitting with a straight line. Given training data tuples $(x^{(i)}, y^{(i)})$, we want to find a straight line that best fits the data. The line is represented by the equation $y = wx + b$, where $w$ is the slope and $b$ is the y-intercept.

$$ y^{(i)} = f_{w,b}(x^{(i)}) = wx^{(i)}+b $$

The goal is to find the best values for $w$ and $b$ such that the line fits the data as closely as possible. We will see how this is done.

## Cost/Loss Function

$$ \mathcal{J}(w,b) = \frac{1}{2m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

Aim:

$$\min\limits_{w,b} \mathcal{J}(w,b)$$

For this we need gradient descent algorithm.

## Gradient Descent
Repeat until convergence:

$$\begin{equation*}\begin{split} w \leftarrow w - \alpha \frac{\partial}{\partial w} \mathcal{J}(w,b) \\
b \leftarrow b - \alpha \frac{\partial}{\partial b} \mathcal{J}(w,b) \end{split}\end{equation*}$$

where $\alpha$ is the learning rate.

> Simultaneous update of $w$ and $b$ is important. So, before update, store values of $\frac{\partial}{\partial w} \mathcal{J}(w,b)$ and $\frac{\partial}{\partial b} \mathcal{J}(w,b)$ in temporary variables.
{: .prompt-info}

Some important points to note:
- If $\alpha$ is too small, gradient descent can be slow.
- If $\alpha$ is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge.
- If $w$ reaches a minimum, then $\frac{\partial}{\partial w} \mathcal{J}(w,b) = 0$ and similarly for $b$, so no change in $w$ and $b$ will occur.

> Can we somehow find $\alpha$ automatically? Yes, we can. It's called learning rate decay. We will see that in the future.
{: .prompt-info}

<details>
<summary> Derivation of $\frac{\partial}{\partial w} \mathcal{J}(w,b)$ and $\frac{\partial}{\partial b} \mathcal{J}(w,b)$ </summary>

$$\begin{equation*}\begin{split}
\frac{\partial}{\partial w} \mathcal{J}(w,b) &= \frac{\partial}{\partial w} \frac{1}{2m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \\
&= \frac{1}{2m} \sum_{i=0}^{m-1} 2(f_{w,b}(x^{(i)}) - y^{(i)}) \frac{\partial}{\partial w} (f_{w,b}(x^{(i)}) - y^{(i)}) \\
&= \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) x^{(i)} \\
\end{split}\end{equation*}$$
Similar, derivation for $\frac{\partial}{\partial b} \mathcal{J}(w,b)$.
</details>

## Gradient Descent for Linear Regression  
Repeat until convergence:

$$\begin{equation*}\begin{split} w \leftarrow w - \alpha \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) x^{(i)} \\
b \leftarrow b - \alpha \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \end{split}\end{equation*}$$

Imp points:
- The cost function $\mathcal{J}(w,b)$ is a convex function, so there is only one minimum. Gradient descent will always converge to the global minimum (if the learning rate is not too large), irrespective of the initial values of $w$ and $b$.
- The reason for that comes from statistics, I will try to include that in the future.
- If the cost function is not convex, then gradient descent may converge to a local minimum, which may not be the global minimum.
- This is called Batch Gradient Descent, because we use all the training examples to update $w$ and $b$ in each iteration.

## Higher Dimensions & Vectorization

| General Notation | Description | Python (if applicable) |
| --- | --- | --- |
| $a$ | scalar |  |
| $\mathbf{a}$ | vector |  |
| $\mathbf{A}$ | matrix |  |
| $\mathbf{X}_{n\times d}$ | training example matrix | `X_train` |
| $\mathbf{y}_{n\times 1}$ | training example targets | `y_train` |
| $\mathbf{x}^{(i)}_{d\times 1}$, $y^{(i)}$ | $i_{th}$ Training Example | `X[i]`, `y[i]` |
| $n$ | number of training examples | `n` |
| $d$ | number of features in each example | `d` |
| $\mathbf{w}$ | parameter: weight | `w` |
| $b$ | parameter: bias | `b` |
| $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ | $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)}+b$ | `f_wb` |

The feature matrix looks like this:

$$\mathbf{X} = 
\begin{pmatrix}
 x^{(0)}_0 & x^{(0)}_1 & \cdots & x^{(0)}_{d-1} \\ 
 x^{(1)}_0 & x^{(1)}_1 & \cdots & x^{(1)}_{d-1} \\
 \vdots \\
 x^{(n-1)}_0 & x^{(n-1)}_1 & \cdots & x^{(n-1)}_{d-1} 
\end{pmatrix}_{n\times d} = 
\begin{pmatrix}
 (\mathbf{x}^{(0)})^T \\ 
 (\mathbf{x}^{(1)})^T \\
    \vdots \\
    (\mathbf{x}^{(m-1)})^T
\end{pmatrix}$$

- $\mathbf{x}^{(i)}_{d\times 1}$ is a vector containing example i, where $\mathbf{x}^{(i)} = (x^{(i)}_0, x^{(i)}_1, \cdots , x^{(i)}_d)^T$.
- $x^{(i)}_j$ is element j in example i. The superscript in parenthesis indicates the training example number while the subscript represents an feature index.

The implementation of the model evaluation for all the training examples can be vectorized as follows:

$$\mathbf{f}_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w}^T \mathbf{x}^{(i)} + b = w_1x^{(i)}_1 + w_2x^{(i)}_2 + \cdots + w_{d-1}x^{(i)}_{d-1} + b$$

The special thing about numpy is that vectorized calculations are very efficient and done in parallel, so it's very fast. It takes advantage of the underlying hardware, which is usually a CPU or a GPU.

Difference between vectorized and non-vectorized implementation:

```python
# Non-vectorized
for i in range(n):
    y_hat[i] = w[0]*X[i][0] + w[1]*X[i][1] + ... + w[d-1]*X[i][d-1] + b

# Vectorized
y_hat = np.dot(X, w) + b
# Here numpy broadcasting is used which we will see a bit later.
```
In above code, `np.dot` is the dot product of two arrays. It uses all cores of the CPU to do the calculation in parallel. Also, it is implemented in C, so it's very fast.

