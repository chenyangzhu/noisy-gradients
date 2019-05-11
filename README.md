# Comparison of Algorithms for Noisy First Order Gradients in Stochastic Accelerated Optimization

FInal Project for IEOR 290, UC Berkeley

Chenyang Zhu

# Abstract
In this project, we study stochastic optimization problems with noisy first order gradient descents. We compare the related algorithms on convergence rate in terms of their bias and variance bounds. Two computational examples are computed for both strongly convex and non-strongly convex case. While current methods do well on strongly convex functions, they do not have necessarily optimal convergence for non-strongly convex functions.

# Some results

### σ=1e-2, n=1000 for l1 constrained OLS
![pics](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/non_sc_1000_3.png)

### σ=1e-1, n=10000 for l1 constrained OLS
![pics](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/non_sc_10000_2.png)


### σ=1e-1, n=10000 for l2 constrained Logistic Regression
![pics](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/10000_2.png)
