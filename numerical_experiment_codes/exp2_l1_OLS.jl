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
    result =  sum((X * β - y) .^ 2) / 2 / n + λ *norm(β,1)
    return result
end

function ∇f(β)
    #=
    This is already noisy gradient.
    β: 100 x 1
    σ: noise
    =#
    return X' * (X * β - y) ./n .+ λ + rand(Normal(0,σ^2), (d,1))
end

# MASG

function MASG(f, ∇f, nk, α, μ)
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
    f_history = zeros(sum(nk))
    γ = 1

    for k in 1:K
        x_0 = x_1
        for m in 1:nk[k]
            w = (1 - sqrt(μ * α[k])) / (1 + sqrt(μ * α[k]))
            ymk = (1 + w) .* x_1 - w .* x_0
            x_0 = x_1
            x_1 = ymk .- α[k] .* ∇f(ymk)

            # Record History
            f_history[γ] = f(x_1)
            γ = γ + 1
        end
    end
    return f_history
end

function compute_nk(κ, p, Δ, label)
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
function GD(f, ∇f)
    x = zeros(d)
    f_history = zeros(η)
    for i in 1:η
        x = x - 1 / L * ∇f(x)
        f_history[i] = f(x)
    end
    return f_history
end

# Accelerated Gradient Descent
function AGD(f, ∇f)
    x_0 = x_1 = zeros(d)
    f_history = zeros(η)
    for i in 1:η
        y = x_1 + (i-1)/(i+2) *(x_1-x_0)
        x_0 = x_1
        x_1 = y - 1/L * ∇f(y)

        f_history[i] = f(x_1)
    end
    return f_history
end

# AC-SA
## Bregman Distance
function D(β1, β2)
    #=
    Prox function, Bregman Distance
    =#
    return f(β1) - f(β2) - (∇f(β2)' * (β1 .- β2))[1]
end

function ORDA(τ)
    #=
    ξ is the turning param
    M is constant
    τ is quadratic constant
    =#
    M = 0
    c = (sqrt(τ) * (σ + M))/(2 * sqrt(D(βstart, zeros(d))) )
    θ(t) = 2/(t+2)
    ν(t) = 2/(t+1)
    γ(t) = c * (t+1)^(3/2)+τ*L
    G = 0
    zt = z0 = xt = x0 = zeros(d)
    f_history = zeros(η+1)
    for t in 0:η
        f_history[t+1] = f(xt)
        p1 = (1 - θ(t)) * (μ + θ(t)^2 * γ(t)) / (θ(t)^2 * γ(t) + (1-θ(t)^2) * μ) .* xt
        p2 = ((1 - θ(t)) * θ(t) * μ + θ(t)^3 * γ(t)) / (θ(t)^2 * γ(t) + (1-θ(t)^2) * μ) * zt
        yt = p1 + p2
        G = G .+ ∇f(yt) ./ ν(t)
        gt = θ(t) * ν(t) .* G

        zp1 = ∇f(x0) .- λ .- (gt .+ λ) ./ (θ(t) * ν(t) * γ(t+1))
        zt = inv(X' * X) * (zp1 .* n + X' * y)

        xp1 = ∇f(yt)  .- λ .- (∇f(yt) .+ λ) / (μ / (τ * θ(t)^2) + γ(t) / τ)
        xt = inv(X' * X) * (xp1 .* n + X' * y)
     end
     return f_history
end

function ACSA(ν)
    α(t) = 2/(t+1)
    γ(t) = 4 * 2 * L / (ν * t * (t+1))

    xt = xtag = zeros(d)
    f_history = zeros(η+1)
    f_history[1] = f(xt)

    for t in 1:η
        xp1 = (1 - α(t)) * (μ + γ(t)) / (γ(t) + (1-α(t)^2) * μ) * xtag
        xp2 = α(t)*((1-α(t)) * μ +γ(t))/(γ(t) + (1-α(t)^2) * μ) * xt
        xtmd = xp1 + xp2

        p1 = (∇f(xtmd) * (μ-1) * α(t) .- α(t) * λ .+ ((1-α(t))*μ + γ(t))*∇f(xt)) / (μ + γ(t))
        p2 = (p1 .- λ) .*n + X' * y
        xt = inv(X' * X) * p2

        xtag = α(t) * xt + (1- α(t)) * xtag
        f_history[t+1] = f(xt)
    end

    return f_history
end



## Data setup
global n = 10000   # number of datapoints
global d = 100    # number of dimensions
global λ = 0.01   # Penalty for beta term
global ρ = 0
global X = rand(Normal(0,1), (n,d))
β = rand(Normal(0, 1), (d, 1))
global βstart = β
global y = X * β  .+ rand(Normal(0,1),(n,1))

# Experiments
global K = 10     # number of iterations for MASG
global η = 2000   # Number of iterations for other methods
global L = maximum(eigvals(X' * X))
# global L = norm(X,2)^2/n
global μ = 0

global σ = sqrt(1e-1)
# MASG
α_MASG = compute_MASG_α(L)
nk_7 = compute_nk(κ, 1, norm(β,2),3.7)
f_MASG_7 = MASG(f, ∇f, nk_7, α_MASG, μ)

## MASG Thm 4
nk_4 = compute_nk(κ, 1, norm(β,2), 4.0)
f_MASG_4 = MASG(f, ∇f, nk_4, α_MASG, μ)

# Vanilla Gradient Descent
f_GD = GD(f, ∇f)

# Vanilla Accelerated Gradient Descent
f_AGD = AGD(f, ∇f)

# ORDA
f_ORDA = ORDA(L)

ν = norm(X,2) / n
f_ACSA = ACSA(ν)

plot(f_GD, label="GD",legend=:bottomleft,w=2,linestyle=:dash,color=:red, xscale = :log, yscale=:log)
plot!(f_AGD, label="AGD", w=2, linestyle=:dash,color=:purple)
plot!(f_ORDA, label = "ORDA", w=2, color=:blue)
plot!(f_ACSA, label="ACSA",w=2, color=:brown)
plot!(f_MASG_7, label="MASG",w=2, linestyle=:solid, color=:green)
plot!(f_MASG_4, label="MASG*",w=2, linestyle=:solid, color=:black)
xlabel!("Iteration")

savefig("./figures/non_sc_10000_4.png")
