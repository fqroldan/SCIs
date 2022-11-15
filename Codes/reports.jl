function param_table(dd::DebtMod)

    τ = ifelse(dd.pars[:τ] <= minimum(dd.gr[:y]), -Inf, dd.pars[:τ])

    rownames = [
        "Sovereign's risk aversion",
        "Interest rate",
        "Income autocorrelation coefficient",
        "Standard deviation of \$y_{t}\$",
        "Preference shock scale parameter",
        "Reentry probability",
        "Duration of debt",
        "Discount factor",
        "Default cost: linear",
        "Default cost: quadratic",
        "Degree of robustness",
        "Linear coupon indexation",
        "Repayment threshold"
    ]

    param_names = [
        "\$\\gamma\$",
        "\$r\$",
        "\$\\rho\$",
        "\$\\sigma_{\\epsilon}\$",
        "\$\\chi\$",
        "\$\\psi\$",
        "\$\\delta\$",
        "\$\\beta\$",
        "\$d_0\$",
        "\$d_1\$",
        "\$\\theta\$",
        "\$\\alpha\$",
        "\$\\tau\$",
    ]

    params = [
        dd.pars[:γ],
        dd.pars[:r],
        dd.pars[:ρy],
        dd.pars[:σy],
        dd.pars[:χ],
        dd.pars[:ψ],
        dd.pars[:ρ],
        dd.pars[:β],
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