function simul_cons(dd::DebtMod, T=150, K=10_000)
    
    ϵvv, ξvv = simulshocks(T, K);

    itp_yield = get_yields_itp(dd);
    itp_qRE, itp_qdRE = q_RE(dd, do_calc = false);
    itp_spr_og = itp_mti(dd, do_calc = false);

    Ny = length(dd.gr[:y]);
    u_bar = Vector{Vector{Float64}}(undef, Ny);
    c_bar = Vector{Vector{Float64}}(undef, Ny);

    for (jy, yv) in enumerate(dd.gr[:y])
        pv, _ = simulvec(dd, itp_yield, itp_qRE, itp_qdRE, itp_spr_og, ϵvv, ξvv, burn_in = 0, stopdef = false, B0 = 0., Y0 = yv)

        c_all = [ifelse(pp[:ζ, t] == 2, pp[:y, t], pp[:c, t]) for t in 1:length(pv[1][:ζ]), pp in pv]

        u_bar[jy] = mean([u(c, dd) for c in c_all], dims=2) |> vec
        c_bar[jy] = mean(c_all, dims = 2) |> vec

    end
    return u_bar, c_bar
end

function NPDV(dd::DebtMod, yvec)
    β = dd.pars[:β]

    T = length(yvec)

    v = 0.0
    for jt in 1:T
        v += β^jt * yvec[jt]
    end

    return v
end

function NPDVu(dd::DebtMod, cvec)
    uvec = [u(c, dd) for c in cvec]
    return NPDV(dd, uvec)
end

function welfare_decomp(dd::DebtMod; T = 150, K = 10_000)

    u_bar, c_bar = simul_cons(dd, T, K)

    w = dd.v[:V][1, :]

    w_c_bar = [NPDVu(dd, c_bar[jy]) for jy in eachindex(dd.gr[:y])]
    w_u_bar = [NPDV(dd, u_bar[jy])  for jy in eachindex(dd.gr[:y])]

    c = [cons_equiv(v, dd) for v in w]
    c_cbar = [cons_equiv(v,dd) for v in w_c_bar]
    c_ubar = [cons_equiv(v,dd) for v in w_u_bar]

    return c, c_ubar, c_cbar
end

function welfare_decomp_at_y(dd::DebtMod; kwargs...)
    c, c_ubar, c_cbar = welfare_decomp(dd; kwargs...)

    jy = ceil(Int, length(dd.gr[:y]) / 2)

    return c[jy], c_ubar[jy], c_cbar[jy]
end

function welfare_decomp_opt(dd::DebtMod, dd_opt::DebtMod; showtop = false)
    @assert dd.pars[:α] == 0 && dd.pars[:τ] <= minimum(dd.gr[:y])
    modelname = ifelse(dd.pars[:θ] > 1e-3, "Benchmark", "Rational Expectations")

    c, c_ubar, c_cbar = welfare_decomp_at_y(dd)

    co, co_ubar, co_cbar = welfare_decomp_at_y(dd_opt)

    names = [
        "Total gains",
        "From default costs",
        "From volatility",
        "From level"
    ]

    qs = [
        co / c,
        (co / co_ubar) / (c / c_ubar),
        (co_ubar / co_cbar) / (c_ubar / c_cbar),
        co_cbar / c_cbar
    ]

    rp = maximum(length(name) for name in names) + 2


    tab = ""
    if showtop
        tab *= rpad(" ", rp, " ")
        for jn in eachindex(names)
            tab *= "& " * rpad(names[jn], rp, " ")
        end
        tab *= "\\\\ \n"
    end
    tab *= rpad(modelname, rp, " ")
    for jn in eachindex(names)
        Q = @sprintf("%0.3g", 100 * (qs[jn] - 1))
        tab *= "& " * rpad(Q, rp, " ")
    end
    tab *= "\\\\ \n"
    
    tab
end
