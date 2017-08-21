module timeevolution2

export master, master_dynamic

import QuantumOptics.timeevolution: recast!
import QuantumOptics: tensor

using QuantumOptics
using QuantumOptics.operators: gemm!
using ..mask2
using ..state2

const DecayRates = QuantumOptics.timeevolution.timeevolution_master.DecayRates


function master(tspan, state0::State, H::LazySum, J::Vector{LazyTensor};
                rates::DecayRates=nothing,
                Jdagger::Vector{LazyTensor}=dagger.(J),
                fout::Union{Function,Void}=nothing,
                kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    D = allocate_memory(state0)
    masks = state0.masks
    H_ = LazySum(H.factors, [TensorOperator(masks, h) for h in H.operators])
    J_ = [TensorOperator(masks, j) for j in J]
    Jdagger_ = [TensorOperator(masks, j) for j in Jdagger]
    function dmaster_(t::Float64, state::State, dstate::State)
        dmaster(state, H_, rates, J_, Jdagger_, dstate, D)
    end
    x0 = Vector{Complex128}(length(state0))
    recast!(state0, x0)
    state = copy(state0)
    dstate = copy(state0)
    QuantumOptics.timeevolution.integrate(tspan_, dmaster_, x0, state, dstate, fout, kwargs...)
end

master(tspan, state0, H::LazyTensor, J; kwargs...) = master(tspan, state0, LazySum(H), J; kwargs...)



function recast!(rho::State, x::Vector{Complex128})
    i = 1
    for op in rho.operators
        Ni = length(op.data)
        copy!(x, i, op.data, 1, Ni)
        i += Ni
    end
    for op in rho.correlations
        Ni = length(op.data)
        copy!(x, i, op.data, 1, Ni)
        i += Ni
    end
    x
end

function recast!(x::Vector{Complex128}, rho::State)
    i = 1
    for op in rho.operators
        Ni = length(op.data)
        copy!(op.data, 1, x, i, Ni)
        i += Ni
    end
    for op in rho.correlations
        Ni = length(op.data)
        copy!(op.data, 1, x, i, Ni)
        i += Ni
    end
    rho
end

function allocate_memory(state::State)
    D = Dict{String, Any}()
    D["tmp_rho1"] = copy(state)
    D["tmp_rho2"] = copy(state)
    correlations = Dict{Mask, Dict{Mask, DenseOperator}}()
    for mask in keys(state.correlations)
        correlations[mask] = Dict{Mask, DenseOperator}()
    end
    D["cachedptrace"] = CachedPTrace(state.N, state.basis_l, state.basis_r,
                                    Mask(state.N),
                                    state.operators,
                                    zeros(Complex128, state.N),
                                    correlations,
                                    Dict{Mask, Complex128}())
    D
end

type PTracePlan
    operator2operator::Vector{Int}
    operator2correlation::Vector{Int}
    correlation2operator::Vector{Int}
    correlation2correlation::Vector{Int}
end


function subtraces(indices::Vector{Int}, operators::Vector{DenseOperator})
    N = length(operators)
    x = ones(Complex128, N)
    for i=indices
        x[i] = trace(operators[i])
    end
    x
end

function tensor{N}(a::AbstractArray, m::Mask{N})
    @assert length(a) == N
    reduce(tensor, a, m)
end

type HamiltonianPlan_rho{N}
    indices_left::Mask{N}
    subtraces_masks::Vector{Mask{N}}
    indices_p_s::Vector{Mask{N}}
    subtraces_masks_p_s::Vector{Vector{Mask{N}}}
    indices_p_s_::Vector{Mask{N}}
    subtraces_masks_p_s_::Vector{Vector{Mask{N}}}
    indices_ptrace::Vector{Vector{Mask{N}}}
end

type HamiltonianPlan_sigma{N}

end

type IncoherentPlan_rho{N}

end

type IncoherentPlan_sigma{N}

end

type MasterPlan
    hplan_rho::Vector{HamiltonianPlan_rho}
    hplan_sigma::Vector{HamiltonianPlan_sigma}
    jplan_rho::Vector{IncoherentPlan_rho}
    jplan_sigma::Vector{IncoherentPlan_sigma}
end

function create_hamiltonianplan{N}(state::State, H_u::LazyTensor)
    m = Mask{N}(H_u.indices)
    L = length(m)
    indices_left = m
    subtraces_masks = Vector{Mask{N}}(L)
    indices_p_s = Vector{Mask{N}}(L)
    subtraces_masks_p_s = Vector{Vector{Mask{N}}}(L)
    indices_p_s_ = Vector{Mask{N}}(L)
    subtraces_masks_p_s_ = Vector{Vector{Mask{N}}}(L)
    for k in 1:length(m)
        i_p = m[k]
        subtraces_masks[k] = setdiff(m, i_p)

    indices_p_s = [i_s for i_s=1:length(state.masks) if state.masks[i_s]]
end

function create_incoherentplan(rho::State, J::Vector{TensorOperator})
    IncoherentPlan_rho(), IncoherentPlan_sigma()
end

function create_masterplan(rho::State, H::LazySum, J::Vector{TensorOperator})
    N = length(H.indices)
    hplans_rho = Vector{HamiltonianPlan_rho}(N)
    hplans_sigma = Vector{HamiltonianPlan_sigma}(N)
    for u=1:length(H.indices)
        hplans_rho[u], hplans_sigma[u] = create_hamiltonianplan(rho, H.operators[u])
    end
    jplan_rho, jplan_h = create_incoherentplan(rho, J)
    MasterPlan(hplan_rho, hplan_sigma, jplan_rho, jplan_sigma)
end

function dmaster{N}(rho::State, H::LazySum,
                 rates,
                 J::Vector{TensorOperator}, Jdagger::Vector{TensorOperator},
                 drho::State, tmp::Dict{String, Any}, plan::MasterPlan{N})
    cache1 = tmp["cache1"]
    cache2 = tmp["cache2"]
    h_rho = tmp["h_rho"]
    rho_h = tmp["rho_h"]

    fill!(drho, 0.)
    drho.factor = rho.factor

    for u in 1:length(H.operators)
        a = H.factors[u]
        h = H.operators[u]
        mul!(-1im*a, h, rho, cache1, h_rho)
        mul!(1im*a, rho, h, cache2, rho_h)

        subtraces_ = subtraces(h.indices, h_rho.operators) # tr(h*rho)

        # into operators
        hplan_rho = plan.hplan_rho[u]
        for k in 1:length(hplan_rho.indices_left) # drho_{p'} = tr_p [h, rho]
            i_p = hplan_rho.indices_left[k]
            for j in 1:length(hplan_rho.indices_p_s_) # p' ∩ s' == {}
                i_s = hplan_rho.indices_p_s_[k][j]
                f2 = prod(subtraces, hplan_rho.innersubtraces_masks[k][j]) # cᵤ ∩ p ∩ s'
                drho.operators[i_p] = -1im*f2*ptrace(h_rho.correlations[i_s] - rho_h.correlations[i_s], plan_rho.indices_ptrace[k][j])
            end
            f = prod(subtraces, hplan_rho.subtraces_masks[k]) # cᵤ ∩ p
            for j in 1:length(hplan_rho.indices_p_s) # p' ∩ s == {}
                i_s = hplan_rho.indices_p_s[k][j]
                f2 = prod(subtraces, hplan_rho.innersubtraces_masks[k][j]) # cᵤ ∩ p ∩ s'
                f += f2*trace(h_rho.correlations[i_s])
            end
            drho.operators[i_p] += -1im*f*(h_rho.operators[i_p] - rho_h.operators[i_p])
        end

        # into correlations
        hplan_sigma = plan.hplan_sigma[u]
        for i_p in
            f = prod(subtraces, cᵤ ∩ p)
            for i_s in
                f2 = prod(subtraces, cᵤ ∩ p ∩ s')
                if p' ∩ s == 0
                    f += f2*trace(h_rho.correlations[i_s])
                elseif p' ∩ s' == 0
                    drho.correlations[i_p] += f2*ptrace(h_rho.correlations[i_s] - rho_h.correlations[i_s], s ∪ p)
                else
                    sigma = ptrace(h_rho.correlations[i_s], s ∩ p)
                    drho.correlations[i_p] += f2*embedcorrelation(s' ∩ p', h_rho.operators, s ∩ p', sigma)
                    sigma = ptrace(rho_h.correlations[i_s], s ∩ p)
                    drho.correlations[i_p] -= f2*embedcorrelation(s' ∩ p', rho_h.operators, s ∩ p', sigma)
                end
            end
            drho.correlations[i_p] += f*(tensor(h_rho.operators, cᵤ ∩ s' ∩ p') - tensor(rho_.operators, cᵤ ∩ p'))
        end
    end
    # _dmaster_J(rho, rates, J, Jdagger, drho, tmp)
    # doperators = drho.operators
    # dcorrelations = drho.correlations
    # for order=2:length(rho.masks)
    #     for mask in rho.masks[order]
    #         I = mask2indices(mask)
    #         suboperators = rho.operators[I]
    #         # Tr{̇̇d/dt ρ}
    #         σ_I = dcorrelations[mask]
    #         # d/dt ρ^{s_k}
    #         for i = 1:order
    #             σ_I -= embedcorrelation(suboperators, [i], doperators[I[i]])
    #         end
    #         # d/dt σ^{s}
    #         for submask in keys(dcorrelations)
    #             if !issubmask(submask, mask)
    #                 continue
    #             end
    #             σ_I -= embedcorrelation(suboperators, submask[I], dcorrelations[submask])
    #             for i in setdiff(mask2indices(complement(submask)), mask2indices(complement(mask)))
    #                 ops = [i==j ? doperators[j] : rho.operators[j] for j in I]
    #                 σ_I -= embedcorrelation(ops, submask[I], rho.correlations[submask])
    #             end
    #         end
    #         dcorrelations[mask] = σ_I
    #     end
    # end
end


end # module
