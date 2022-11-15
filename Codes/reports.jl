function param_table(dd::DebtMod)

    τ = ifelse(dd.pars[:τ] <= minimum(dd.gr[:y]), -Inf, dd.pars[:τ])

    rownames = [
        "Sovereign's discount factor",
        "Sovereign's risk aversion",
        "Preference shock scale parameter",
        "Interest rate",
        "Duration of debt",
        "Average coupon rate",
        "Income autocorrelation coefficient",
        "Standard deviation of \$y_{t}\$",
        "Reentry probability",
        "Default cost: linear",
        "Default cost: quadratic",
        "Degree of robustness",
        "Linear coupon indexation",
        "Repayment threshold"
    ]

    param_names = [
        "\$\\beta\$",
        "\$\\gamma\$",
        "\$\\chi\$",
        "\$r\$",
        "\$\\delta\$",
        "\$\\bar{\\kappa}\$",
        "\$\\rho\$",
        "\$\\sigma_{\\epsilon}\$",
        "\$\\psi\$",
        "\$d_0\$",
        "\$d_1\$",
        "\$\\theta\$",
        "\$\\alpha\$",
        "\$\\tau\$",
    ]

    params = [
        dd.pars[:β],
        dd.pars[:γ],
        dd.pars[:χ],
        dd.pars[:r],
        dd.pars[:ρ],
        dd.pars[:κ],
        dd.pars[:ρy],
        dd.pars[:σy],
        dd.pars[:ψ],
        dd.pars[:d1],
        dd.pars[:d2],
        dd.pars[:θ],
        dd.pars[:α],
        τ
    ]

    textpad = maximum(length(name) for name in rownames) + 2
    parpad = maximum(length(par) for par in param_names) + 2
    
    table = ""
    for jp in eachindex(rownames)
        table *= rpad(rownames[jp], textpad, " " ) * "& "
        table *= rpad(param_names[jp], parpad, " ") * "& "
        table *= "$(@sprintf("%0.4g",params[jp]))"
        table *= "   \\\\\n"
    end
    return table
end

function save_param_table(dd; filename="table3.txt")
    table = param_table(dd)

    write(filename, table)
end