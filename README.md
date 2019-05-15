# Comparison of Algorithms for Noisy First Order Gradients in Stochastic Accelerated Optimization

FInal Project for IEOR 290, Spring 2019, UC Berkeley

Chenyang Zhu

# Abstract
In this project, we study stochastic optimization problems with noisy first order gradient descents. We compare the related algorithms on convergence rate in terms of their bias and variance bounds. Two computational examples are computed for strongly convex and non-strongly convex cases to compare methods that are designed for optimal convergence. We also applied the flexible step-size method proposed in Multistage Accelerated Stochastic Gradient Descent (MASG) to Accelerated Stochastic Approximation Algorithm (ACSA). Our results show that while the stabilization in MASG seems to work well, it is not yet a universal method and that a more rigorous proof might be necessary before applying this idea to other methods.

Download paper [here](https://github.com/chenyangzhu/noisy-gradients/raw/master/Chenyang%20Zhu%2C%20Comparison%20of%20Algorithms%20for%20Noisy%20First%20Order%20Gradients%20in%20Stochastic%20Accelerated%20Optimization%2C%202019.pdf)

# Computational Results

|  | σ2=1e-1  | σ2=1e-2  |  σ2=1e-3 |
|---|---|---|---|
| l2-logistic, 1k | ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/1000_4.png)  | ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/1000_2.png) | ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/1000_3.png)|
| l2-logistic, 10k  | ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/10000_4.png)   |  ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/10000_2.png)  |  ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/1000_3.png)  |
| l1-linear, 1k  |  ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/non_sc_1000_4.png) |  ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/non_sc_1000_2.png) |![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/non_sc_1000_3.png) |
| l1-linear, 10k  |  ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/non_sc_10000_4.png) |  ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/non_sc_10000_2.png) |![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/non_sc_10000_3.png) |
| MASG to ACSA  |  ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/ACSA_new_4.png) | ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/ACSA_new_1.png)  | ![](https://github.com/chenyangzhu/noisy-gradients/raw/master/figures/ACSA_new_2.png)|
