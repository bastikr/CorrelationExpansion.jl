# CorrelationExpansion.jl

**CorrelationExpansion.jl** is a library that can be used to simulate large open quantum systems. By specifying which quantum correlations should be included and which should be neglected the dimensionality of the problem can be reduced to a tractable size. Then a time evolution according to a master equation in this approximation can be performed completely automatically.
This package is written in [Julia] and builds on [QuantumOptics.jl].


## Development status

  * Linux/Mac build: [![Travis build status][travis-img]][travis-url]
  * Windows build: [![Windows build status][appveyor-img]][appveyor-url]
  * Test coverage:
        [![Test coverage status on coveralls][coveralls-img]][coveralls-url]
        [![Test coverage status on codecov][codecov-img]][codecov-url]


## Installation

**CorrelationExpansion.jl** is not an officially registered package but it nevertheless can be installed using julia's package manager:

```julia
julia> Pkg.clone("https://github.com/bastikr/CorrelationExpansion.jl.git")
```

## Documentation

The documentation can be found at

https://bastikr.github.io/CorrelationExpansion.jl/latest


[Julia]: http://julialang.org
[qojulia]: https://github.com/qojulia
[QuantumOptics.jl]: https://bastikr.github.io/QuantumOptics.jl

[travis-url]: https://travis-ci.org/bastikr/CorrelationExpansion.jl
[travis-img]: https://api.travis-ci.org/bastikr/CorrelationExpansion.jl.png?branch=master

[appveyor-url]: https://ci.appveyor.com/project/bastikr/correlationexpansion-jl/branch/master
[appveyor-img]: https://ci.appveyor.com/api/projects/status/heib5o43485r90uq/branch/master?svg=true

[coveralls-url]: https://coveralls.io/github/bastikr/CorrelationExpansion.jl?branch=master
[coveralls-img]: https://coveralls.io/repos/github/bastikr/CorrelationExpansion.jl/badge.svg?branch=master

[codecov-url]: https://codecov.io/gh/bastikr/CorrelationExpansion.jl
[codecov-img]: https://codecov.io/gh/bastikr/CorrelationExpansion.jl/branch/master/graph/badge.svg
