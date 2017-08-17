module state2

export State, correlation, approximate, CachedPTrace, add_into, embedcorrelation, complement, TensorOperator, mul!

import Base: trace, ==, +, -, *, /
import QuantumOptics: dagger, identityoperator,
            ptrace, normalize!, tensor, permutesystems
import QuantumOptics.operators: gemm!, check_ptrace_arguments

using QuantumOptics, Combinatorics
using ..mask2

const sortedindices = QuantumOptics.sortedindices
import QuantumOptics.sortedindices: complement
complement(x::AbstractArray{Bool}) = [!i for i=x]

function ptrace2(state, indices)
    if length(indices)==length(basis(state).bases)
        return trace(state)
    else
        return ptrace(state, indices)
    end
end


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
    masks::Vector{Mask}
    correlations::Vector{DenseOperator}
end

function State(masks::Vector{Mask}, operators::Vector{DenseOperator},
            correlations::Vector{DenseOperator},
            factor::Number=1)
    N = length(operators)
    basis_l = CompositeBasis(Basis[op.basis_l for op in operators])
    basis_r = CompositeBasis(Basis[op.basis_r for op in operators])
    State(N, basis_l, basis_r, factor, copy.(operators), masks, copy.(correlations))
end

function State(masks::Vector{Mask}, operators::Vector{DenseOperator}, factor::Number=1)
    N = length(operators)
    basis_l = CompositeBasis(Basis[op.basis_l for op in operators])
    basis_r = CompositeBasis(Basis[op.basis_r for op in operators])
    correlations = DenseOperator[]
    for m in masks
        push!(correlations, DenseOperator(CompositeBasis(basis_l.bases[m]...), CompositeBasis(basis_r.bases[m]...)))
    end
    State(N, basis_l, basis_r, factor, copy.(operators), masks, correlations)
end

type TensorOperator <: Operator
    N::Int
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    factor::Complex128
    indices::Vector{Int}
    operators::Vector{Operator}
    indices_correlations::Vector{Int}
    operators_correlations::Vector{Operator}
end

function TensorOperator(masks::Vector{Mask}, op::LazyTensor)
    N = length(op.basis_l.bases)
    indices_correlations = Int[]
    operators_correlations = Operator[]
    m_op = Mask(N, op.indices)
    for i=1:length(masks)
        m_cor = masks[i]
        m_both = m_op ∩ m_cor
        if length(m_both.indices) > 0
            bl = CompositeBasis(op.basis_l.bases[m_cor]...)
            br = CompositeBasis(op.basis_r.bases[m_cor]...)
            push!(indices_correlations, i)
            push!(operators_correlations, embed(bl, br, m_both[m_cor].indices, operators_lazytensor.suboperators(op, m_both.indices)))
        end
    end
    TensorOperator(N, op.basis_l, op.basis_r, op.factor,
                    op.indices, op.operators,
                    indices_correlations, operators_correlations)
end

function mul!(alpha, op::TensorOperator, x::State, cache::State, result::State)
    result.factor = op.factor*alpha*x.factor
    beta = complex(0.)
    alpha = complex(1.)
    n = 1
    N_op = length(op.indices)
    for i=1:x.N
        if n <= N_op && i == op.indices[n]
            gemm!(alpha, op.operators[n], x.operators[i], beta, cache.operators[i])
            result.operators[i] = cache.operators[i]
            n += 1
        else
            result.operators[i] = x.operators[i]
        end
    end
    n = 1
    N_op = length(op.indices_correlations)
    for i=1:length(x.masks)
        if n <= N_op && i == op.indices_correlations[n]
            gemm!(alpha, op.operators_correlations[n], x.correlations[i], beta, cache.correlations[i])
            result.correlations[i] = cache.correlations[i]
            n += 1
        else
            result.correlations[i] = x.correlations[i]
        end
    end
end

function mul!(alpha, x::State, op::TensorOperator, cache::State, result::State)
    result.factor = op.factor*alpha*x.factor
    beta = complex(0.)
    alpha = complex(1.)
    n = 1
    N_op = length(op.indices)
    for i=1:x.N
        if n <= N_op && i == op.indices[n]
            gemm!(alpha, x.operators[i], op.operators[n], beta, cache.operators[i])
            result.operators[i] = cache.operators[i]
            n += 1
        else
            result.operators[i] = x.operators[i]
        end
    end
    n = 1
    N_op = length(op.indices_correlations)
    for i=1:length(x.masks)
        if n <= N_op && i == op.indices_correlations[n]
            gemm!(alpha, x.correlations[i], op.operators_correlations[n], beta, cache.correlations[i])
            result.correlations[i] = cache.correlations[i]
            n += 1
        else
            result.correlations[i] = x.correlations[i]
        end
    end
end


# function State(basis_l::CompositeBasis, basis_r::CompositeBasis, masks, factor::Number=1)
#     N = length(basis_l.bases)
#     operators = [DenseOperator(basis_l.bases[i], basis_r.bases[i]) for i=1:N]
#     correlations = Vector{DenseOperator}(length(masks))
#     for i=1:length(masks)
#         m = masks[i]
#         @assert length(m) > 1
#         correlations[m] = tensor(operators[m]...)
#     end
#     State(operators, correlations)
# end
# State(basis::CompositeBasis, masks) = State(basis, basis, masks)
# function State(operators::Vector, masks)
#     N = length(operators)
#     correlations = Dict{Mask, DenseOperator}()
#     for op in operators
#         @assert typeof(op) == DenseOperator
#     end
#     for m in masks
#         m = as_mask(N, m)
#         @assert sum(m) > 1
#         b_l = CompositeBasis([op.basis_l for op in operators[m]]...)
#         b_r = CompositeBasis([op.basis_r for op in operators[m]]...)
#         correlations[m] = DenseOperator(b_l, b_r)
#     end
#     State(operators, correlations)
# end

function Base.length(x::State)
    L = sum(Int[length(op.basis_l)*length(op.basis_r) for op in x.operators])
    L += sum(Int[length(op.basis_l)*length(op.basis_r) for op in x.correlations])
    L
end

function Base.copy(x::State)
    State(x.N, x.basis_l, x.basis_r, x.factor, copy.(x.operators), x.masks, copy.(x.correlations))
end

"""
    embedcorrelation(ops, mask, correlation)

Tensor product of a correlation and the density operators of the remaining subsystems.

# Arguments
* `operators`: Vector containing the reduced density operators of each subsystem.
* `mask`: Indices specifying on which subsystems the given
        correlation is defined.
* `correlation`: Correlation operator defined in the specified subsystems.
"""
function embedcorrelation(operators::Vector{DenseOperator}, mask::Mask,
            correlation::DenseOperator)
    imin = mask.indices[1]
    imax = mask.indices[end]
    left_operators = operators[1:imin-1]
    right_operators = operators[imax+1:end]
    if imax-imin+1 == length(mask.indices)
        op = correlation
    else
        indices_cor = mask
        indices_ops = [i for i in imin:imax if i ∉ mask.indices]
        perm = sortperm([indices_cor.indices; indices_ops])
        mixed_ops = operators[indices_ops]
        op = permutesystems(tensor(correlation, mixed_ops...), perm)
    end
    tensor(left_operators..., op, right_operators...)
end
# function embedcorrelation(operators::Vector{DenseOperator}, mask::Mask,
#             correlation::Number)
#     @assert sum(mask) == 0
#     tensor(operators...)*correlation
# end
# embedcorrelation(operators::Vector{DenseOperator}, indices::Vector{Int}, correlation) = embedcorrelation(operators, indices2mask(length(operators), indices), correlation)

# """
#     correlation(rho, mask; <keyword arguments>)

# Calculate the correlation of the subsystems specified by the given mask.

# # Arguments
# * `rho`: Density operator of the total system.
# * `mask`: Indices or mask specifying on which subsystems the given
#         correlation is defined.
# * `operators` (optional):  A tuple containing all reduced density
#         operators of the single subsystems.
# * `subcorrelations`: A (mask->operator) dictionary storing already
#         calculated correlations.
# """
# function correlation(rho::DenseOperator, mask::Mask;
#             operators::Vector{DenseOperator}=(N=length(mask); [ptrace2(normalize(rho), complement(N, [i])) for i in 1:N]),
#             subcorrelations::Dict{Mask, DenseOperator}=Dict{Mask, DenseOperator}())
#     # Check if this correlation was already calculated.
#     @assert length(mask) == length(rho.basis_l.bases)
#     if mask in keys(subcorrelations)
#         return subcorrelations[mask]
#     end
#     order = sum(mask)
#     rho = normalize(rho)
#     σ = ptrace2(rho, mask2indices(complement(mask)))
#     σ -= tensor(operators[mask]...)
#     for submask in subcorrelationmasks(mask)
#         subcorrelation = correlation(rho, submask;
#                                      operators=operators,
#                                      subcorrelations=subcorrelations)
#         σ -= embedcorrelation(operators[mask], submask[mask], subcorrelation)
#     end
#     subcorrelations[mask] = σ
#     σ
# end
# correlation(rho::Operator, indices::Vector{Int}) = correlation(rho, indices2mask(length(rho.basis_l.bases), indices))
# correlation(rho::State, mask::Mask) = rho.correlations[mask]

# """
#     approximate(rho[, masks])

# Correlation expansion of a density operator.

# If masks are specified, only these correlations are included.
# """
# function approximate(rho::DenseOperator, masks)
#     @assert typeof(rho.basis_l) == CompositeBasis
#     N = length(rho.basis_l.bases)
#     alpha = trace(rho)
#     rho = normalize(rho)
#     operators = [ptrace2(rho, complement(N, [i])) for i in 1:N]
#     subcorrelations = Dict{Mask, DenseOperator}() # Dictionary to store intermediate results
#     correlations = Dict{Mask, DenseOperator}()
#     for m in masks
#         m = as_mask(N, m)
#         correlations[m] = correlation(rho, m;
#                                          operators=operators,
#                                          subcorrelations=subcorrelations)
#     end
#     State(operators, correlations, alpha)
# end
# function approximate(rho::DenseOperator)
#     @assert typeof(rho.basis_l) == CompositeBasis
#     N = length(rho.basis_l.bases)
#     approximate(rho, Set{Mask}())
# end


ptrace2(mask::Mask, indices::Vector{Int}) = mask[complement(length(mask), indices)]

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
            correlation = factor*ptrace2(rho.correlations[mask], J)
        end
        op = embedcorrelation(operators, ptrace2(mask, indices), correlation)
        result += op
    end
    rho.factor*result
end

function Base.full(rho::State)
    result = tensor(rho.operators...)
    for i=1:length(rho.masks)
        result += embedcorrelation(rho.operators, rho.masks[i], rho.correlations[i])
    end
    rho.factor*result
end


function Base.fill!(state::State, a::Number)
    for op in state.operators
        fill!(op.data, a)
    end
    for correlation in state.correlations)
        fill!(correlation.data, a)
    end
    state.factor = 1.
end

end # module
