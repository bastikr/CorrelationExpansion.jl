using Base.Test

using CorrelationExpansion2
const ce = CorrelationExpansion2

@testset "mask" begin

m = ce.Mask(4, [1, 3])

@test [2:6;][m] == [2, 4]

@test m[[1,2]] == ce.Mask(2, [1])
@test m[ce.Mask(4, [2,3])] == ce.Mask(2, [2])
@test m[[1,3,4]] == ce.Mask(3, [1,2])

@test ce.complement(m) == ce.Mask(4, [2, 4])

S2 = ce.masks(4, 2)
S3 = ce.masks(4, 3)
S4 = ce.masks(4, 4)
@test S2 == [ce.Mask(4, a) for a in ([1,2], [1,3], [1,4], [2,3], [2,4], [3,4])]
@test S3 == [ce.Mask(4, a) for a in ([1,2,3], [1,2,4], [1,3,4], [2,3,4])]
@test S4 == [ce.Mask(4, [1,2,3,4])]

@test ce.masks(4) == S2 ∪ S3 ∪ S4

@test ce.submasks(ce.Mask(4, [1,3,4])) == [ce.Mask(4, a) for a in ([1,3], [1,4], [3,4])]

@test ce.Mask(5, [2, 4, 5]) ∩ ce.Mask(5, [2, 3]) == ce.Mask(5, [2])
@test ce.Mask(5, [1, 3]) ∩ ce.Mask(5, [1, 3, 4]) == ce.Mask(5, [1, 3])

@test ce.Mask(5, [3, 4]) ⊆ ce.Mask(5, [1, 3, 4])
@test !(ce.Mask(5, [3, 4]) ⊆ ce.Mask(5, [1, 3]))

end # testset
