module timeevolution

export master, master_dynamic

import QuantumOptics.timeevolution: recast!

using QuantumOptics
using QuantumOptics.operators: gemm!
using ..mask
using ..state

const DecayRates = QuantumOptics.timeevolution.timeevolution_master.DecayRates


function master(tspan, state0::State, H::LazySum, J::Vector{LazyTensor};
                rates::DecayRates=nothing,
                Jdagger::Vector{LazyTensor}=dagger.(J),
                fout::Union{Function,Void}=nothing,
                kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    D = allocate_memory(state0)
    function dmaster_(t::Float64, state::State, dstate::State)
        dmaster(state, H, rates, J, Jdagger, dstate, D)
    end
    x0 = Vector{Complex128}(length(state0))
    recast!(state0, x0)
    state = copy(state0)
    dstate = copy(state0)
    QuantumOptics.timeevolution.integrate(tspan_, dmaster_, x0, state, dstate, fout; kwargs...)
end

master(tspan, state0, H::LazyTensor, J; kwargs...) = master(tspan, state0, LazySum(H), J; kwargs...)


function master_dynamic(tspan, state0::State, f::Function;
                rates::DecayRates=nothing,
                fout::Union{Function,Void}=nothing,
                kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    D = allocate_memory(state0)
    function dmaster_dynamic_(t::Float64, state::State, dstate::State)
        dmaster_dynamic(t, state, f, rates, dstate, D)
    end
    x0 = Vector{Complex128}(length(state0))
    recast!(state0, x0)
    state = copy(state0)
    dstate = copy(state0)
    QuantumOptics.timeevolution.integrate(tspan_, dmaster_dynamic_, x0, state, dstate, fout; kwargs...)
end

function recast!(rho::State, x::Vector{Complex128})
    i = 1
    for op in rho.operators
        Ni = length(op.data)
        copy!(x, i, op.data, 1, Ni)
        i += Ni
    end
    for masks_n in rho.masks
        for mask in masks_n
            op = rho.correlations[mask]
            Ni = length(op.data)
            copy!(x, i, op.data, 1, Ni)
            i += Ni
        end
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
    for masks_n in rho.masks
        for mask in masks_n
            op = rho.correlations[mask]
            Ni = length(op.data)
            copy!(op.data, 1, x, i, Ni)
            i += Ni
        end
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


function _dmaster_J(rho::State,
                 rates::Void,
                 J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor},
                 drho::State, tmp::Dict{String, Any})
    cache = tmp["cachedptrace"]
    tmp1 = tmp["tmp_rho1"]
    tmp2 = tmp["tmp_rho2"]
    for i=1:length(J)
        gemm!(1, J[i], rho, complex(0.), tmp2)
        gemm!(complex(1.), tmp2, Jdagger[i], complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(complex(-0.5), Jdagger[i], tmp2, complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(-0.5, rho, Jdagger[i], complex(0.), tmp2)
        gemm!(complex(1.), tmp2, J[i], complex(0.), tmp1)
        add_into(tmp1, drho, cache)
    end
end

function _dmaster_J(rho::State,
                 rates::Vector{Float64},
                 J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor},
                 drho::State, tmp::Dict{String, Any})
    cache = tmp["cachedptrace"]
    tmp1 = tmp["tmp_rho1"]
    tmp2 = tmp["tmp_rho2"]
    for i=1:length(J)
        gemm!(rates[i], J[i], rho, complex(0.), tmp2)
        gemm!(complex(1.), tmp2, Jdagger[i], complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(complex(-0.5), Jdagger[i], tmp2, complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(-0.5*rates[i], rho, Jdagger[i], complex(0.), tmp2)
        gemm!(complex(1.), tmp2, J[i], complex(0.), tmp1)
        add_into(tmp1, drho, cache)
    end
end

function _dmaster_J(rho::State,
                 rates::Matrix{Float64},
                 J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor},
                 drho::State, tmp::Dict{String, Any})
    cache = tmp["cachedptrace"]
    tmp1 = tmp["tmp_rho1"]
    tmp2 = tmp["tmp_rho2"]
    for j=1:length(J), i=1:length(J)
        gemm!(rates[i,j], J[i], rho, complex(0.), tmp2)
        gemm!(complex(1.), tmp2, Jdagger[j], complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(complex(-0.5), Jdagger[j], tmp2, complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(-0.5*rates[i,j], rho, Jdagger[j], complex(0.), tmp2)
        gemm!(complex(1.), tmp2, J[i], complex(0.), tmp1)
        add_into(tmp1, drho, cache)
    end
end

function dmaster(rho::State, H::LazySum,
                 rates,
                 J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor},
                 drho::State, tmp::Dict{String, Any})
    cache = tmp["cachedptrace"]
    tmp1 = tmp["tmp_rho1"]
    tmp2 = tmp["tmp_rho2"]
    fill!(drho, 0.)
    drho.factor = rho.factor
    for i in 1:length(H.operators)
        a = H.factors[i]
        h = H.operators[i]
        gemm!(-1im*a, h, rho, complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(1im*a, rho, h, complex(0.), tmp1)
        add_into(tmp1, drho, cache)
    end
    _dmaster_J(rho, rates, J, Jdagger, drho, tmp)
    doperators = drho.operators
    dcorrelations = drho.correlations
    for order=2:length(rho.masks)
        for mask in rho.masks[order]
            I = mask2indices(mask)
            suboperators = rho.operators[I]
            # Tr{̇̇d/dt ρ}
            σ_I = dcorrelations[mask]
            # d/dt ρ^{s_k}
            for i = 1:order
                σ_I -= embedcorrelation(suboperators, [i], doperators[I[i]])
            end
            # d/dt σ^{s}
            for submask in keys(dcorrelations)
                if !issubmask(submask, mask)
                    continue
                end
                σ_I -= embedcorrelation(suboperators, submask[I], dcorrelations[submask])
                for i in setdiff(mask2indices(complement(submask)), mask2indices(complement(mask)))
                    ops = [i==j ? doperators[j] : rho.operators[j] for j in I]
                    σ_I -= embedcorrelation(ops, submask[I], rho.correlations[submask])
                end
            end
            dcorrelations[mask] = σ_I
        end
    end
end

function dmaster_dynamic(t::Float64, state::State, f::Function, rates::DecayRates,
                dstate::State, tmp::Dict{String, Any})
    result = f(t, state)
    @assert 3 <= length(result) <= 4
    if length(result) == 3
        H, J, Jdagger = result
        rates_ = rates
    else
        H, J, Jdagger, rates_ = result
    end
    dmaster(state, H, rates_, J, Jdagger, dstate, tmp)
end

end # module
