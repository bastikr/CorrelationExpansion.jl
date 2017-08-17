using Base.Test

using QuantumOptics
using CorrelationExpansion2


@testset "state" begin

ce = CorrelationExpansion2
srand(0)
D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))

b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(2)
b4 = GenericBasis(7)
b = tensor(b1, b2, b3, b4)

h1 = randoperator(b1)
h2 = randoperator(b2)
h3 = randoperator(b3)
h4 = randoperator(b4)

psi1 = randstate(b1)
psi2 = randstate(b2)
psi3 = randstate(b3)
psi4 = randstate(b4)

rho1 = dm(psi1)
rho2 = dm(psi2)
rho3 = dm(psi3)
rho4 = dm(psi4)

ops = [rho1, rho2, rho3, rho4]

masks = ce.masks(4, 3)
state = ce.State(masks, ops)
h = LazyTensor(b, [2, 3], [h2, h3])
h_ = ce.TensorOperator(masks, h)

cache = ce.State(masks, ops)
result = ce.State(masks, ops)

ce.mul!(1., h_, state, cache, result)
@test 1e-10 > D(result, rho1 ⊗ (h2*rho2) ⊗ (h3*rho3) ⊗ rho4)

ce.mul!(1., state, h_, cache, result)
@test 1e-10 > D(result, rho1 ⊗ (rho2*h2) ⊗ (rho3*h3) ⊗ rho4)
# state = ce.State()

end # testset
