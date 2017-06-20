using Base.Test
using QuantumOptics
using CorrelationExpansion

@testset "correlationexpansion" begin

ce = CorrelationExpansion

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))

b1 = FockBasis(1)
b2 = SpinBasis(1//2)
b3 = NLevelBasis(2)
b4 = NLevelBasis(2)
b = tensor(b1, b2, b3, b4)

randdo(b) = (x = randstate(b); x ⊗ dagger(x))

# Test Masks
mask = ce.indices2mask(3, [1,2])
@test mask == BitArray([1, 1, 0])
indices = ce.mask2indices(mask)
@test indices == [1,2]

S2 = ce.masks(4, 2)
S3 = ce.masks(4, 3)
S4 = ce.masks(4, 4)
@test S2 == Set([[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]])
@test S3 == Set([[1,2,3], [1,2,4], [1,3,4], [2,3,4]])
@test S4 == Set([[1,2,3,4]])

# Test correlation calculation
rho = randdo(b1⊗b2)
sigma = ce.correlation(rho, [1,2])
@test 1e-13 > D(sigma, rho - ptrace(rho, 2) ⊗ ptrace(rho, 1))

# Test creation of ApproximateOperator
op1 = randoperator(b)
op1_ = ce.approximate(op1, S2 ∪ S3 ∪ S4)
@test 1e-13 > D(op1, op1_)

# Test multiplication
h = 0.5*LazyTensor(b, [1, 2, 3, 4], [randoperator(b1), randoperator(b2), randoperator(b3), randoperator(b4)])

@test 1e-13 > D(full(h)*0.3*full(op1_), h*(0.3*op1_))
@test 1e-13 > D(full(op1_)*0.3*full(h), (op1_*0.3)*h)
@test 1e-13 > D(full(h)*full(op1_)*0.3*full(h), h*(op1_*0.3)*h)

# Test ptrace
x_ = h*op1_
x = full(h)*op1
@test 1e-13 > D(ptrace(x, 1), ptrace(x_, 1))
@test 1e-13 > D(ptrace(x, 2), ptrace(x_, 2))
@test 1e-13 > D(ptrace(x, 3), ptrace(x_, 3))
@test 1e-13 > D(ptrace(x, 4), ptrace(x_, 4))
@test 1e-13 > D(ptrace(x, [1,2]), ptrace(x_, [1,2]))
@test 1e-13 > D(ptrace(x, [1,3]), ptrace(x_, [1,3]))
@test 1e-13 > D(ptrace(x, [1,4]), ptrace(x_, [1,4]))
@test 1e-13 > D(ptrace(x, [2,3]), ptrace(x_, [2,3]))
@test 1e-13 > D(ptrace(x, [2,4]), ptrace(x_, [2,4]))
@test 1e-13 > D(ptrace(x, [3,4]), ptrace(x_, [3,4]))
@test 1e-13 > D(ptrace(x, [1,2,3]), ptrace(x_, [1,2,3]))
@test 1e-13 > D(ptrace(x, [1,2,4]), ptrace(x_, [1,2,4]))
@test 1e-13 > D(ptrace(x, [1,3,4]), ptrace(x_, [1,3,4]))
@test 1e-13 > D(ptrace(x, [2,3,4]), ptrace(x_, [2,3,4]))

# Compare to standard master time evolution
rho = randdo(b1) ⊗ randdo(b2) ⊗ randdo(b3) ⊗ randdo(b4)
rho_ce = ce.approximate(rho, S2 ∪ S3 ∪ S4)

j1 = LazyTensor(b, [1, 2, 3, 4], [randoperator(b1), randoperator(b2), randoperator(b3), randoperator(b4)])
j2 = LazyTensor(b, [1, 2, 3, 4], [randoperator(b1), randoperator(b2), randoperator(b3), randoperator(b4)])
J = LazyTensor[j1, j2]
v = rand(Float64, length(J))
Γ = v * transpose(v)

H = LazySum(h, dagger(h))


T = [0.:0.005:0.01;]
tout_ce, rho_ce_t = ce.master(T, rho_ce, H, J; rates=Γ)

tout, rho_t = timeevolution.master_h(T, full(rho), full(H), [full(j) for j in J]; rates=Γ)
for i=1:length(rho_t)
    @test 1e-5 > D(rho_ce_t[i], rho_t[i])
end

end # testset
