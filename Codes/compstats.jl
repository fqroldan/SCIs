function comp_argbond(dd::DebtMod; show_simul=false, othersimuls=false, DEP=false)
    @assert dd.pars[:α] == 0 && dd.pars[:τ] <= minimum(dd.gr[:y])

    mpe_simul!(dd, maxiter=500, K=8, simul=false)

    Random.seed!(25)
    w, t, m, ϵvv, ξvv = calib_targets(dd, cond_K=7_500, uncond_K=10_000, smalltable=true)


    Ny = length(dd.gr[:y])
    v_noncont = dd.v[:V][1, ceil(Int, Ny / 2)]
    c_noncont = cons_equiv(v_noncont, dd)
    print("V with noncont: $v_noncont, c = $c_noncont\n")

    if DEP
        DEP_bylength(dd)
    end

    dd.pars[:α] = 1
    dd.pars[:min_q] = 0

    mpe_simul!(dd, maxiter=500, K=8, simul=false)
    calib_targets(dd, ϵvv, ξvv, uncond_K=10_000, smalltable=true)

    v_linear = dd.v[:V][1, ceil(Int, Ny / 2)]
    c_linear = cons_equiv(v_linear, dd)
    print("V with linear: $v_linear, c = $c_linear. ")
    gains_linear = c_linear / c_noncont - 1
    print("Gains with linear = $(@sprintf("%0.3g", 100*gains_linear))%\n")

    if DEP
        DEP_bylength(dd)
    end

    dd.pars[:τ] = 1
    dd.gr[:b] = collect(range(0, 2 * maximum(dd.gr[:b]), length=length(dd.gr[:b])))

    mpe_simul!(dd, maxiter=500, K=8, simul=false)
    calib_targets(dd, ϵvv, ξvv, uncond_K=10_000, smalltable=true)

    v_threshold = dd.v[:V][1, ceil(Int, Ny / 2)]
    c_threshold = cons_equiv(v_threshold, dd)
    print("V with threshold: $v_threshold, c = $c_threshold. ")
    gains_threshold = c_threshold / c_noncont - 1
    print("Gains with threshold = $(@sprintf("%0.3g", 100*gains_threshold))%\n")
    if DEP
        DEP_bylength(dd)
    end

    return v_noncont, v_linear, v_threshold
end

function comp_t5(dd::DebtMod)
    @assert dd.pars[:θ] >= 1e-3

    print("Solving original model with θ = $(dd.pars[:θ])\n")
    save("dd_comp_theta.jld2", "dd", dd)
    rob_n, rob_l, rob_t = comp_argbond(dd, show_simul=true, othersimuls=true, DEP = true)

    dd = load("dd_comp_theta.jld2", "dd")
    dd.pars[:θ] = 0
    print("Solving same model with rational expectations\n")
    rat_n, rat_l, rat_t = comp_argbond(dd, show_simul=true, othersimuls=true)

    return rob_n, rob_l, rob_t, rat_n, rat_l, rat_t
end

function DEP_bylength(dd::DebtMod, Yvec = range(10, 60, length=6))

    for years in Yvec
        year = Int(years)
        _, DEP = simul_dist(dd, K = 2_000, burn_in = 2_000, T = 4year)
        print("DEP computed on $year years ($(4year) periods): $(@sprintf("%0.3g", 100*DEP))%\n")
    end
end