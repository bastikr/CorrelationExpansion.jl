var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "CorrelationExpansion.jl",
    "title": "CorrelationExpansion.jl",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#CorrelationExpansion.jl-1",
    "page": "CorrelationExpansion.jl",
    "title": "CorrelationExpansion.jl",
    "category": "section",
    "text": ""
},

{
    "location": "api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api.html#CorrelationExpansion.state.State",
    "page": "API",
    "title": "CorrelationExpansion.state.State",
    "category": "Type",
    "text": "State(operators, correlations[, factor])\nState(operators, masks)\nState(basis, masks)\nState(basis_l, basis_r, masks)\n\nState representing a correlation expansion series.\n\nPhysically the state consists of a product state  and the selected correlations . Which correlations are included is specified by the given masks s ∈ Sₙ.\n\n =  + _s  S tilde\n\n\n\n"
},

{
    "location": "api.html#CorrelationExpansion.state.correlation",
    "page": "API",
    "title": "CorrelationExpansion.state.correlation",
    "category": "Function",
    "text": "correlation(rho, mask; <keyword arguments>)\n\nCalculate the correlation of the subsystems specified by the given mask.\n\nArguments\n\nrho: Density operator of the total system.\nmask: Indices or mask specifying on which subsystems the given       correlation is defined.\noperators (optional):  A tuple containing all reduced density       operators of the single subsystems.\nsubcorrelations: A (mask->operator) dictionary storing already       calculated correlations.\n\n\n\n"
},

{
    "location": "api.html#CorrelationExpansion.state.embedcorrelation",
    "page": "API",
    "title": "CorrelationExpansion.state.embedcorrelation",
    "category": "Function",
    "text": "embedcorrelation(ops, mask, correlation)\n\nTensor product of a correlation and the density operators of the remaining subsystems.\n\nArguments\n\noperators: Vector containing the reduced density operators of each subsystem.\nmask: Indices or mask specifying on which subsystems the given       correlation is defined.\ncorrelation: Correlation operator defined in the specified subsystems.\n\n\n\n"
},

{
    "location": "api.html#CorrelationExpansion.state.approximate",
    "page": "API",
    "title": "CorrelationExpansion.state.approximate",
    "category": "Function",
    "text": "approximate(rho[, masks])\n\nCorrelation expansion of a density operator.\n\nIf masks are specified, only these correlations are included.\n\n\n\n"
},

{
    "location": "api.html#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "CorrelationExpansion.StateCorrelationExpansion.correlationCorrelationExpansion.embedcorrelationCorrelationExpansion.approximate"
},

]}
