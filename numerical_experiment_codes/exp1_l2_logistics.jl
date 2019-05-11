using LinearAlgebra
using Random, Distributions
using Plots
Plots.scalefontsizes(1.5)


# Global Terms
Random.seed!(1234)


# Logistic Regression

function f(β)
    #=
    Objective Function
    x: n x d
    y: d x 1
    β: d x 1
    λ: penalty
    =#
    neg_yxβ = - X * β .* y
    log_term = log.(1 .+ exp.(neg_yxβ))
    # log_term = min.(log_term, 10000)  # Prevent Overflow
    summation = sum(log_term) / n
    return summation + λ/2 * norm(β,2)^2
end

function ∇f(β, σ)
    #=
    This is already noisy gradient.
    β: 100 x 1
    σ: noise
    =#
    neg_yxβ =  - X * β .* y
    weight = exp.(neg_yxβ) ./ (1 .+ exp.(neg_yxβ))

    # Replace nan in weight
    for idx in findall(isnan, weight)
        weight[idx] = 0
    end

    sums = sum(weight .* (-X .* y),dims=1) / n
    # @show sums
    # @show size(sums)
    gradient = sums' + λ .* β
    return gradient + rand(Normal(0,σ^2), d)
end
# MASG

function MASG(f, ∇f, nk, α, μ, σ)
    #=
        f:  objective functin
        ∇f: gradient of f
        X:  design matrix
        y:  {-1,1} class
        a_n:an array containing each step each iteration
        K:  total number of iterations
        α:  an array containing step sizes
        μ:  ⁠μ-strongly convex
        λ:  penalty
    =#
    x_0 = x_1 = zeros(d)

    # Record Histories
    x_history = zeros(sum(nk))
    f_history = zeros(sum(nk))
    γ = 1

    for k in 1:K
        x_0 = x_1
        for m in 1:nk[k]
            w = (1 - sqrt(μ * α[k])) / (1 + sqrt(μ * α[k]))
            x_updated = (1 + w) .* x_1 - w .* x_0
            fx_updated = f(x_updated)
            x_0 = x_1
            x_1 = x_updated .- α[k] .* ∇f(x_updated, σ)


            # Record History
            x_history[γ] = norm(x_1)
            f_history[γ] = f(x_1)
            γ = γ + 1
        end
    end
    return x_history, f_history
end

function compute_nk(κ, p, Δ, σ, label)
    C = floor(sqrt(κ) * log(2^(p+2))) + 1
    grid = [1:K;]
    nk = 2 .^ grid
    if label == 3.7
        nk[1] = floor(sqrt(κ) * log(κ)) + 1
    elseif label == 4
        nk[1] = floor(sqrt(κ) * log(2 * L * Δ / (σ^2 * sqrt(κ)))) + 1
    end
    return nk
end

function compute_MASG_α(L)
    grid = [1:K;]
    α = 1 ./ (2 .^ (2 .* [1:K;]) .* L)
    α[1] = 1 / L
    return α
end

# Vanilla Gradient Descent
function GD(f, ∇f, σ)
    x = zeros(d)
    f_history = zeros(η)
    x_history = zeros(η)
    for i in 1:η
        x = x - 1 / L * ∇f(x, σ)
        f_history[i] = f(x)
        x_history[i] = norm(x,2)
    end
    return x_history, f_history
end

# Accelerated Gradient Descent
function AGD(f, ∇f, σ)
    x_0 = x_1 = zeros(d)
    f_history = zeros(η)
    x_history = zeros(η)
    for i in 1:η
        y = x_1 + (i-1)/(i+2) *(x_1-x_0)
        x_0 = x_1
        x_1 = y - 1/L * ∇f(y, σ)

        f_history[i] = f(x_1)
        x_history[i] = norm(x_1,2)
    end
    return x_history, f_history
end

# AC-SA
## Bregman Distance
function D(β1, β2)
    #=
    Prox function, Bregman Distance
    =#
    return f(β1) - f(β2) - ∇f(β2) * (β1 - β2)
end

# μAGD

function compute_μAGD_a()
    grid = [1:η;]
    return 1 ./ (κ .* sqrt.(grid))
end

function compute_μAGD_A(a)
    A = zeros(η)
    A[1] = a[1]
    for i in 2:η
        A[i] = A[i-1] + a[i]
    end
    return A
end

function μAGD(f, ∇f, A, a, σ, mode)
    #=
    Mode is for selecting restart + slow down algorithm
    if 0, use vanilla
    if 2, use restart and slowdown-2
    =#
    f_history = zeros(η)
    x_history = zeros(η)

    xk = x0 = zeros(d)
    summation = a[1] .* ∇f(xk,σ)
    yk = x0 .- summation / λ
    f_history[1] = f(xk)
    a_square_sum = a[1]^2
    z0 = λ .* xk
    changed_flag = false

    for k in 2:η
        xk = A[k-1] / A[k] .* yk + a[k] / A[k] .* (x0 .- summation ./ λ)
        summation = summation .+ a[k] .* ∇f(xk, σ)
        yk = A[k-1] / A[k] .* yk + a[k] ./ A[k] .* (x0 .- summation ./ λ)
        zk = - summation .+ z0
        a_square_sum += a[k]^2

        if mode == 2 && norm(zk,2) <= (a_square_sum * d) && changed_flag == false
            a = compute_μAGD_a()
            A = compute_μAGD_A(a)
            changed_flag == true
        end

        # Record history
        f_history[k] = f(xk)
        x_history[k] = norm(xk, 2)
    end
    return x_history, f_history
end

## Data setup
global n = 10000    # number of datapoints
global d = 100    # number of dimensions
global λ = 0.01   # Penalty for beta term
global X = rand(Normal(0,1), (n,d))
β = rand(Normal(0, 1), (d, 1))
global y = sign.(X * β)

# Experiments
global K = 12      # number of iterations for MASG
global η = 8000   # Number of iterations for other methods
global κ = 1000   # condition number from the paper
global L = 0.25 * maximum(eigvals(X' * X))
global μ = L / κ

σ = sqrt(1e-1)

# MASG
α_MASG = compute_MASG_α(L)
## MASG Thm 3.7
nk_7 = compute_nk(κ, 1, norm(β,2), σ, 3.7)
x_MASG_7, f_MASG_7 = MASG(f, ∇f, nk_7, α_MASG, μ, σ)
## MASG Thm 4
nk_4 = compute_nk(κ, 1, norm(β,2), σ, 4.0)
x_MASG_4, f_MASG_4 = MASG(f, ∇f, nk_4, α_MASG, μ, σ)

# Vanilla Gradient Descent
x_GD, f_GD = GD(f, ∇f, σ)

# Vanilla Accelerated Gradient Descent
x_AGD, f_AGD = AGD(f, ∇f, σ)

# μAGD+
μAGD_a = compute_μAGD_a()
μAGD_A = compute_μAGD_A([1:η;] ./ L)
x_μAGD_0, f_μAGD_0 = μAGD(f,∇f,μAGD_A,[1:η;] ./ L,σ,0)
x_μAGD_2, f_μAGD_2 = μAGD(f,∇f,μAGD_A,[1:η;] ./ L,σ,2)


plot(f_GD, label="GD",legend=:topleft, xscale = :log, yscale = :log,w=2,linestyle=:solid,color=:red)
plot!(f_AGD, label="AGD", w=2, linestyle=:solid,color=:purple)
plot!(f_μAGD_0, label= "AGD+", xscale = :log, yscale = :log, color=:blue,w=2,linestyle=:dashdot)
plot!(f_μAGD_2, label= "AGD+v.2", xscale = :log, yscale = :log, color=:grey,w=2,linestyle=:dashdot)
plot!(f_MASG_7, label="MASG",w=2, linestyle=:dash, color=:green)
plot!(f_MASG_4, label="MASG*",w=2, linestyle=:dash, color=:black)
xlabel!("Iteration")
ylabel!("f")
savefig("./figures/10000_4.png")
