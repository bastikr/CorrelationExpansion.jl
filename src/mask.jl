module mask

export Mask, indices2mask, mask2indices, as_mask, masks, subcorrelationmasks, issubmask

using QuantumOptics, Combinatorics, Iterators

const sortedindices = QuantumOptics.sortedindices

typealias Mask BitArray{1}

function indices2mask(N::Int, indices::Vector{Int})
    m = Mask(N)
    for i in indices
        m[i] = true
    end
    m
end

function indices2mask(N::Int, indices::Vector{Int}, m::Mask)
    fill!(m, false)
    for i in indices
        m[i] = true
    end
    m
end

mask2indices(mask::Mask) = find(mask)

as_mask(N::Int, m::Mask) = (@assert length(m)==N; m)
function as_mask(N::Int, m)
    m = collect(m)
    sortedindices.check_indices(N, m)
    indices2mask(N, m)
end

masks(N::Int, order::Int) = Set(combinations(1:N, order))
masks(N::Int) = reduce(∪, [masks(N, order) for order=2:N])

subcorrelationmasks(mask::Mask) = [indices2mask(length(mask), indices) for indices in
        chain([combinations(mask2indices(mask), k) for k=2:sum(mask)-1]...)]

function issubmask(submask::Mask, mask::Mask)
    if sum(submask) >= sum(mask)
        return false
    end
    for i=1:length(mask)
        if submask[i] && !mask[i]
            return false
        end
    end
    true
end

end # module
