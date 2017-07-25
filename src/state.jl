module state

export State, correlation, approximate, CachedPTrace, add_into, embedcorrelation, complement

import Base: trace, ==, +, -, *, /
import QuantumOptics: dagger, identityoperator,
            ptrace, normalize!, tensor, permutesystems
import QuantumOptics.operators: gemm!, check_ptrace_arguments

using QuantumOptics, Combinatorics
using ..mask

const sortedindices = QuantumOptics.sortedindices
import QuantumOptics.sortedindices: complement
complement(x::AbstractArray{Bool}) = [!i for i=x]


"""
    State(operators, correlations[, factor])
    State(operators, masks)
    State(basis, masks)
    State(basis_l, basis_r, masks)

State representing a correlation expansion series.

Physically the state consists of a product state ``ρ₁`` and the selected
correlations ``σˢ``. Which correlations are included is specified by
the given masks `s ∈ Sₙ`.
```math
ρ = ρ₁ + ∑_{s ∈ Sₙ} \\tilde{σ}ˢ
```
"""
type State <: Operator
    N::Int
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    factor::Complex128
    operators::Vector{DenseOperator}
    masks::Vector{Vector{Mask}}
    correlations::Dict{Mask, DenseOperator}
end

function State(operators::Vector{DenseOperator},
            correlations::Dict{Mask, DenseOperator},
            factor::Number=1)
    N = length(operators)
    basis_l = CompositeBasis(Basis[op.basis_l for op in operators])
    basis_r = CompositeBasis(Basis[op.basis_r for op in operators])
    maxorder = 1
    for (mask, op) in correlations
        @assert length(mask) == N
        @assert sum(mask) > 1
        @assert op.basis_l == tensor(basis_l.bases[mask]...)
        @assert op.basis_r == tensor(basis_r.bases[mask]...)
        maxorder = max(sum(mask), maxorder)
    end
    masks = Vector{Mask}[Mask[] for i=1:maxorder]
    for (mask, op) in correlations
        push!(masks[sum(mask)], mask)
    end
    ops = [copy(op) for op in operators]
    cors = Dict{Mask, DenseOperator}(m=>copy(op) for (m, op) in correlations)
    State(N, basis_l, basis_r, factor, ops, masks, cors)
end

function State(basis_l::CompositeBasis, basis_r::CompositeBasis, masks)
    N = length(basis_l.bases)
    operators = [DenseOperator(basis_l.bases[i], basis_r.bases[i]) for i=1:N]
    correlations = Dict{Mask, DenseOperator}()
    for m in masks
        m = as_mask(N, m)
        @assert sum(m) > 1
        correlations[m] = tensor(operators[m]...)
    end
    State(operators, correlations)
end
State(basis::CompositeBasis, masks) = State(basis, basis, masks)
function State(operators::Vector, masks)
    N = length(operators)
    correlations = Dict{Mask, DenseOperator}()
    for op in operators
        @assert typeof(op) == DenseOperator
    end
    for m in masks
        m = as_mask(N, m)
        @assert sum(m) > 1
        b_l = CompositeBasis([op.basis_l for op in operators[m]]...)
        b_r = CompositeBasis([op.basis_r for op in operators[m]]...)
        correlations[m] = DenseOperator(b_l, b_r)
    end
    State(operators, correlations)
end

function Base.length(x::State)
    L = sum(Int[length(op.basis_l)*length(op.basis_r) for op in x.operators])
    L += sum(Int[length(op.basis_l)*length(op.basis_r) for op in values(x.correlations)])
    L
end

function Base.copy(x::State)
    correlations = Dict(mask=>copy(op) for (mask, op) in x.correlations)
    State(x.N, x.basis_l, x.basis_r, x.factor, copy.(x.operators), x.masks, correlations)
end

"""
    embedcorrelation(ops, mask, correlation)

Tensor product of a correlation and the density operators of the remaining subsystems.

# Arguments
* `operators`: Vector containing the reduced density operators of each subsystem.
* `mask`: Indices or mask specifying on which subsystems the given
        correlation is defined.
* `correlation`: Correlation operator defined in the specified subsystems.
"""
function embedcorrelation(operators::Vector{DenseOperator}, mask::Mask,
            correlation::DenseOperator)
    imin = findfirst(mask)
    imax = findlast(mask)
    left_operators = operators[1:imin-1]
    right_operators = operators[imax+1:end]
    J = mask[imin:imax]
    if all(J)
        op = correlation
    else
        indices_cor = [i for i in imin:imax if mask[i]]
        indices_ops = [i for i in imin:imax if !mask[i]]
        perm = sortperm([indices_cor; indices_ops])
        mixed_ops = operators[[i for i in imin:imax if !mask[i]]]
        op = permutesystems(tensor(correlation, mixed_ops...), perm)
    end
    tensor(left_operators..., op, right_operators...)
end
function embedcorrelation(operators::Vector{DenseOperator}, mask::Mask,
            correlation::Number)
    @assert sum(mask) == 0
    tensor(operators...)*correlation
end
embedcorrelation(operators::Vector{DenseOperator}, indices::Vector{Int}, correlation) = embedcorrelation(operators, indices2mask(length(operators), indices), correlation)

"""
    correlation(rho, mask; <keyword arguments>)

Calculate the correlation of the subsystems specified by the given mask.

# Arguments
* `rho`: Density operator of the total system.
* `mask`: Indices or mask specifying on which subsystems the given
        correlation is defined.
* `operators` (optional):  A tuple containing all reduced density
        operators of the single subsystems.
* `subcorrelations`: A (mask->operator) dictionary storing already
        calculated correlations.
"""
function correlation(rho::DenseOperator, mask::Mask;
            operators::Vector{DenseOperator}=(N=length(mask); [ptrace(normalize(rho), complement(N, [i])) for i in 1:N]),
            subcorrelations::Dict{Mask, DenseOperator}=Dict{Mask, DenseOperator}())
    # Check if this correlation was already calculated.
    @assert length(mask) == length(rho.basis_l.bases)
    if mask in keys(subcorrelations)
        return subcorrelations[mask]
    end
    order = sum(mask)
    rho = normalize(rho)
    σ = ptrace(rho, mask2indices(complement(mask)))
    σ -= tensor(operators[mask]...)
    for submask in subcorrelationmasks(mask)
        subcorrelation = correlation(rho, submask;
                                     operators=operators,
                                     subcorrelations=subcorrelations)
        σ -= embedcorrelation(operators[mask], submask[mask], subcorrelation)
    end
    subcorrelations[mask] = σ
    σ
end
correlation(rho::Operator, indices::Vector{Int}) = correlation(rho, indices2mask(length(rho.basis_l.bases), indices))
correlation(rho::State, mask::Mask) = rho.correlations[mask]

"""
    approximate(rho[, masks])

Correlation expansion of a density operator.

If masks are specified, only these correlations are included.
"""
function approximate(rho::DenseOperator, masks)
    @assert typeof(rho.basis_l) == CompositeBasis
    N = length(rho.basis_l.bases)
    alpha = trace(rho)
    rho = normalize(rho)
    operators = [ptrace(rho, complement(N, [i])) for i in 1:N]
    subcorrelations = Dict{Mask, DenseOperator}() # Dictionary to store intermediate results
    correlations = Dict{Mask, DenseOperator}()
    for m in masks
        m = as_mask(N, m)
        correlations[m] = correlation(rho, m;
                                         operators=operators,
                                         subcorrelations=subcorrelations)
    end
    State(operators, correlations, alpha)
end
function approximate(rho::DenseOperator)
    @assert typeof(rho.basis_l) == CompositeBasis
    N = length(rho.basis_l.bases)
    approximate(rho, Set{Mask}())
end


ptrace(mask::Mask, indices::Vector{Int}) = mask[complement(length(mask), indices)]

function ptrace(rho::State, indices::Vector{Int})
    check_ptrace_arguments(rho, indices)
    N = rho.N
    operators = rho.operators[complement(N, indices)]
    factors = [trace(op) for op in rho.operators]
    result = tensor(operators...)*prod(factors[indices])
    for mask in keys(rho.correlations)
        I = mask2indices(mask)
        factor = prod(factors[setdiff(indices, I)])
        if isempty(I ∩ indices)
            correlation = factor*rho.correlations[mask]
        else
            J = [i-sum(complement(N, I).<i) for i in I ∩ indices]
            correlation = factor*ptrace(rho.correlations[mask], J)
        end
        op = embedcorrelation(operators, ptrace(mask, indices), correlation)
        result += op
    end
    rho.factor*result
end

function Base.full(rho::State)
    result = tensor(rho.operators...)
    for (mask, correlation) in rho.correlations
        result += embedcorrelation(rho.operators, mask, correlation)
    end
    rho.factor*result
end

function *(op1::State, op2::LazyTensor)
    QuantumOptics.bases.check_multiplicable(op1, op2)
    result = copy(op1)
    gemm!(Complex(1.), op1, op2, Complex(0.), result)
    return result
end
function *(op1::LazyTensor, op2::State)
    QuantumOptics.bases.check_multiplicable(op1, op2)
    result = copy(op2)
    gemm!(Complex(1.), op1, op2, Complex(0.), result)
    return result
end
*(a::Number, b::State) = State(b.operators, b.correlations, a*b.factor)
*(a::State, b::Number) = State(a.operators, a.correlations, a.factor*b)

function reduced_indices(I, I_)
    Int[findfirst(j->i==j, I) for i in I_]
end

function gemm!(alpha, a::LazyTensor, b::State, beta, result::State)
    N = b.N
    result.factor = a.factor*alpha*b.factor
    @assert beta == complex(0.)
    alpha = complex(1.)
    n = 1
    Na = length(a.indices)
    for i=1:N
        if n <= Na && i == a.indices[n]
            gemm!(alpha, a.operators[n], b.operators[i], beta, result.operators[i])
            n += 1
        else
            copy!(result.operators[i].data, b.operators[i].data)
        end
    end
    for mask in keys(b.correlations)
        I = mask2indices(mask)
        I_ = sortedindices.intersect(I, a.indices)
        op = result.correlations[mask]
        if isempty(I_)
            copy!(op.data, b.correlations[mask].data)
        else
            operators = operators_lazytensor.suboperators(a, I_)
            sortedindices.reducedindices!(I_, I)
            a_ = LazyTensor(op.basis_l, op.basis_r, I_, operators)
            gemm!(alpha, a_, b.correlations[mask], beta, op)
        end
    end
end

function gemm!(alpha, a::State, b::LazyTensor, beta, result::State)
    N = a.N
    result.factor = a.factor*alpha*b.factor
    @assert beta == complex(0.)
    alpha = complex(1.)
    n = 1
    Nb = length(b.indices)
    for i=1:N
        if n <= Nb && i == b.indices[n]
            gemm!(alpha, a.operators[i], b.operators[n], beta, result.operators[i])
            n += 1
        else
            copy!(result.operators[i].data, a.operators[i].data)
        end
    end
    for mask in keys(a.correlations)
        I = mask2indices(mask)
        I_ = sortedindices.intersect(I, b.indices)
        op = result.correlations[mask]
        if isempty(I_)
            copy!(op.data, a.correlations[mask].data)
        else
            operators = operators_lazytensor.suboperators(b, I_)
            sortedindices.reducedindices!(I_, I)
            b_ = LazyTensor(op.basis_l, op.basis_r, I_, operators)
            gemm!(alpha, a.correlations[mask], b_, beta, op)
        end
    end
end


type CachedPTrace
    N::Int
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    tmpmask::Mask
    operators::Vector{DenseOperator}
    operators_trace::Vector{Complex128}
    correlations::Dict{Mask, Dict{Mask, DenseOperator}}
    correlations_trace::Dict{Mask, Complex128}
end

function muladd(result::Matrix{Complex128}, factor::Complex128, op::Matrix{Complex128})
    for I in eachindex(result)
        result[I] += factor*op[I]
    end
end

function ptrace_into(op::CachedPTrace, I::Mask, indices::Vector{Int},
                    complement_indices::Vector{Int},
                    factor::Complex128, result::DenseOperator)
    N = op.N
    dims = prod(result.basis_l.shape)
    productfactor = factor
    for i in indices
        productfactor *= op.operators_trace[i]
    end
    ops = op.operators[complement_indices]
    for I_σ in keys(op.correlations)
        indices_σ = mask2indices(I_σ)
        complement_indices_σ = complement(N, indices_σ)
        factor_ = factor
        for i in indices
            if i ∉ indices_σ
                factor_ *= op.operators_trace[i]
            end
        end
        if all(I[indices_σ])
            productfactor += factor_*op.correlations_trace[I_σ]
        elseif all(I[complement_indices_σ])
            tmp = op.correlations[I_σ][complement(I[I_σ])].data
            muladd(result.data, factor_, tmp)
        else
            tmp = embedcorrelation(ops, I_σ[complement(I)], op.correlations[I_σ][complement(I[I_σ])]).data
            muladd(result.data, factor_, tmp)
        end
    end
    tmp = tensor(ops...).data
    muladd(result.data, productfactor, tmp)
end

function cache_ptrace_into(rho::State, cache::CachedPTrace)
    N = rho.N
    correlations = cache.correlations
    correlations_trace = cache.correlations_trace
    cache.operators = rho.operators
    for i=1:N
        cache.operators_trace[i] = trace(rho.operators[i])
    end
    for (I_σ, σ) in rho.correlations
        dims = σ.basis_l.shape
        order = sum(I_σ)
        C = cache.correlations[I_σ]
        C[trues(order)] = σ
        for n=order-1:-1:1
            for indices in combinations(1:order, n)
                J = indices2mask(order, indices)
                i = complement(order, indices)[indmin(dims[complement(J)])]
                J_sup = copy(J)
                J_sup[i] = true
                C[J] = ptrace(C[J_sup], sum(J_sup[1:i]))
            end
        end
        J_sup = falses(order)
        J_sup[indmin(dims)] = true
        correlations_trace[I_σ] = trace(C[J_sup])
    end
end

function add_into(op::State, result::State, cache::CachedPTrace)
    N = op.N
    cache_ptrace_into(op, cache)
    indices = [2:N;]
    mask = cache.tmpmask
    indices2mask(N, indices, mask)
    ptrace_into(cache, mask, indices, [1], op.factor, result.operators[1])
    for i=2:N
        indices[i-1] = i-1
        indices2mask(N, indices, mask)
        ptrace_into(cache, mask, indices, [i], op.factor, result.operators[i])
    end
    for (I_σ, σ) in op.correlations
        complement_indices = mask2indices(I_σ)
        indices = complement(N, complement_indices)
        indices2mask(N, indices, mask)
        ptrace_into(cache, mask, indices, complement_indices, op.factor, result.correlations[I_σ])
    end
end

function Base.fill!(rho::State, a::Number)
    for op in rho.operators
        fill!(op.data, a)
    end
    for σ in values(rho.correlations)
        fill!(σ.data, a)
    end
    rho.factor = 1.
end

end # module
