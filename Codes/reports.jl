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

function calib_table(dd::DebtMod, dd_RE::DebtMod; uncond_K=2_000, uncond_burn=2_000, uncond_T=4_000, cond_T=2_000, cond_K=1_000, longrun = false)
    @assert dd_RE.pars[:θ] == 0

    ϵvv_unc, ξvv_unc = simulshocks(uncond_T, uncond_K)
    ϵvv, ξvv = simulshocks(cond_T, cond_K)

    itp_yield = get_yields_itp(dd)
    itp_qRE, itp_qdRE = q_RE(dd, do_calc=false)
    itp_spr_og = itp_mti(dd, do_calc=false)

    pv_uncond, _ = simulvec(dd, itp_yield, itp_qRE, itp_qdRE, itp_spr_og, ϵvv_unc, ξvv_unc, burn_in=uncond_burn, stopdef=false)
    pv, _ = simulvec(dd, itp_yield, itp_qRE, itp_qdRE, itp_spr_og, ϵvv, ξvv)

    _, DEP = simul_dist(dd)
    
    itp_yield_RE = get_yields_itp(dd_RE)
    itp_qRE, itp_qdRE = q_RE(dd_RE, do_calc=false)
    itp_spr_og = itp_mti(dd, do_calc=false)

    pv_uncond_RE, _ = simulvec(dd_RE, itp_yield_RE, itp_qRE, itp_qdRE, itp_spr_og, ϵvv_unc, ξvv_unc, burn_in=uncond_burn, stopdef=false)
    pv_RE, _ = simulvec(dd_RE, itp_yield_RE, itp_qRE, itp_qdRE, itp_spr_og, ϵvv, ξvv)

    if longrun
        return table_moments_with_DEP(pv_uncond, pv_uncond, DEP, pv_uncond_RE, pv_uncond_RE)
    else
        table_moments_with_DEP(pv, pv_uncond, DEP, pv_RE, pv_uncond_RE)
    end
end

function add_to_table(moments, sym)
    value = moments[sym]
    if sym == :DEP
        if isnan(value)
            val = "-"
        else
            val = @sprintf("%0.3g", value) * "\\%"
        end
    else
        val = @sprintf("%0.3g", value)
    end
    return rpad(val, 10, " ")
end

function table_moments_with_DEP(pv::Vector{SimulPath}, pv_uncond::Vector{SimulPath}, DEP, pv_RE=[], pv_uncond_RE=[])

    syms = [:mean_spr,
        # :mean_sp,
        # :sp_RE, :sp_MTI,
        :std_spr, :debt_gdp, :def_prob, :rel_vol, :corr_yc, :corr_ytb, :corr_ysp, :DEP]

    targets = get_targets()

    moments = compute_moments(pv)
    # Number of defaults divided total periods with market access (ζ = 1)
    moments[:def_prob] = compute_defprob(pv_uncond)
    moments[:DEP] = 100 * DEP

    # moments[:mean_spr] = mean(mean(pp[:spread]) for pp in pv_uncond) * 1e4

    if length(pv_RE) > 0
        moments_RE = compute_moments(pv_RE)
        moments_RE[:def_prob] = compute_defprob(pv_uncond_RE)
        moments_RE[:DEP] = NaN
    end

    names = ["Spread (bps)",
        # "Spread OG",
        # "o/w Spread RE", "Spread MTI",
        "Std Spread", "Debt-to-GDP (\\%)", "Default Prob", "Std(c)/Std(y)", "Corr(y,c)", "Corr(y,tb/y)", "Corr(y,spread)", "DEP"]
    maxn = maximum(length(name) for name in names)

    title = "\\toprule \n"
    title *= ("$(rpad("", maxn+2, " "))")
    title *= ("& $(rpad("Data", 10, " "))")
    title *= ("& $(rpad("Benchmark", 10, " "))")
    if length(pv_RE) > 0
        title *= ("& $(rpad("Rational Expectations", 10, " "))")
    end
    title *= ("\\\\ \\midrule\n")
    table = ""
    for (jn, nv) in enumerate(names)

        table *= "$(rpad(nv, maxn+2, " "))"
        table *= "& $(add_to_table(targets, syms[jn]))"
        table *= "& $(add_to_table(moments, syms[jn]))"
        if length(pv_RE) > 0
            table *= "& $(add_to_table(moments_RE, syms[jn]))"
        end
        if jn == 4
            table *= "\\\\\\midrule \n & \\multicolumn{3}{c}{Other moments} \\\\\\midrule \n"
        else
            table *= ("\\\\\n")
        end
    end
    table *= "\\bottomrule"
    table
end
