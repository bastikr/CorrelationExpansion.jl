module mask2

export Mask, masks, submasks, complement, subseteq

using QuantumOptics, Combinatorics, Iterators

const sortedindices = QuantumOptics.sortedindices
import Base: getindex, ==, ⊆, intersect
import QuantumOptics.sortedindices: complement


type Mask
    N::Int
    indices::Vector{Int}
end

Base.copy(m::Mask) = Mask(m.N, copy(m.indices))
Base.length(m::Mask) = length(m.indices)

function ==(a::Mask, b::Mask)
    @assert a.N == b.N
    a.indices == b.indices
end

Base.getindex(a::AbstractArray, m::Mask) = getindex(a, m.indices)
function Base.getindex(m::Mask, indices::Vector{Int})
    _indices = Int[]
    for i=1:length(indices)
        if indices[i] ∈ m.indices
            push!(_indices, i)
        end
    end
    Mask(length(indices), _indices)
end
function Base.getindex(m1::Mask, m2::Mask)
    @assert m1.N == m2.N
    getindex(m1, m2.indices)
end

function Base.conj(m::Mask)
    indices = sortedindices.complement(m.N, m.indices)
    Mask(m.N, indices)
end

function Base.intersect(m1::Mask, m2::Mask)
    @assert m1.N == m2.N
    Mask(m1.N, sortedindices.intersect(m1.indices, m2.indices))
end

function Base.union(m1::Mask, m2::Mask)
end


function masks(N::Int, order)
    collect(Mask(N, indices) for indices in combinations(1:N, order))
end

masks(N::Int) = reduce(∪, [masks(N, order) for order=2:N])

submasks(mask::Mask) = [Mask(mask.N, indices) for indices in chain([combinations(mask.indices, k) for k=2:length(mask.indices)-1]...)]

function ⊆(submask::Mask, mask::Mask)
    @assert submask.N == mask.N
    for i in submask.indices
        if i ∉ mask.indices
            return false
        end
    end
    true
end

end # module
