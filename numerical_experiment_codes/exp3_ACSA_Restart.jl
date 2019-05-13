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


function ACSA_modified(ν,t_empirical)

    α(p) = 2/(p+1)
    γ(p) = 4 * 2 * L / (ν * p * (p+1))
    xt = xtag = zeros(d)
    f_history = zeros(η+1)
    f_history[1] = f(xt)

    for t in 1:t_empirical
        xp1 = (1 - α(t)) * (μ + γ(t)) / (γ(t) + (1-α(t)^2) * μ) * xtag
        xp2 = α(t)*((1-α(t)) * μ +γ(t))/(γ(t) + (1-α(t)^2) * μ) * xt
        xtmd = xp1 + xp2

        p1 = (∇f(xtmd) * (μ-1) * α(t) .- α(t) * λ .+ ((1-α(t))*μ + γ(t))*∇f(xt)) / (μ + γ(t))
        p2 = (p1 .- λ) .*n + X' * y
        xt = inv(X' * X) * p2

        xtag = α(t) * xt + (1- α(t)) * xtag
        f_history[t+1] = f(xt)

        if t % 100 == 0
            @show t
        end
    end
    γ_1(p) = 4 * L / (ν * p * (p+1)) # + (2*10)/(ν+sqrt(p))
    for t in t_empirical+1:η
        xp1 = (1 - α(t)) * (μ + γ_1(t)) / (γ_1(t) + (1-α(t)^2) * μ) * xtag
        xp2 = α(t)*((1-α(t)) * μ +γ_1(t))/(γ_1(t) + (1-α(t)^2) * μ) * xt
        xtmd = xp1 + xp2

        p1 = (∇f(xtmd) * (μ-1) * α(t) .- α(t) * λ .+ ((1-α(t))*μ + γ_1(t))*∇f(xt)) / (μ + γ_1(t))
        p2 = (p1 .- λ) .*n + X' * y
        xt = inv(X' * X) * p2

        xtag = α(t) * xt + (1- α(t)) * xtag
        f_history[t+1] = f(xt)

        if t % 100 == 0
            @show t
        end
    end

    return f_history
end



## Data setup
global n = 2000   # number of datapoints
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

global σ = 1e-1

ν = norm(X,2) / n
f_ACSA = ACSA(ν)
f_ACSA_modified_300 = ACSA_modified(ν,300)
# f_ACSA_modified_500 = ACSA_modified(ν,500)
#f_ACSA_modified_5000 = ACSA_modified(ν,5000)
f_ACSA_modified_0 = ACSA_modified(ν,0)
# f_ACSA_modified_500 = ACSA_modified(ν,50)

plot(f_ACSA, label="ACSA",w=2, color=:red,xscale=:log,yscale=:log,legend=:bottomleft)
plot!(f_ACSA_modified_300, label="ACSA+300",w=2, color=:blue)
plot!(f_ACSA_modified_0, label="ACSA+0",w=2, color=:orange)


savefig("./figures/ACSA_new_1.png")


plot!(f_ACSA_modified_500, label="ACSA+500",w=2, color=:green)
plot!(f_ACSA_modified_5000, label="ACSA+1000",w=2, color=:purple)
# plot!(f_ACSA_modified_500, label="ACSA+500",w=2, color=:orange)



γ(p) = 4 * 2 * L / (ν * p * (p+1))
γ_1(p) = 4 * L / (ν * p * (p+1))#  + (2*10)/(ν+sqrt(p))

gridline = [1:2000;]
plot(γ.(gridline),xscale=:log,yscale=:log)
plot!(γ_1.(gridline))
