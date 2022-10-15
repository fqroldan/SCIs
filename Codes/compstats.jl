function comp_argbond(dd::DebtMod; show_simul=false, DEP=false)
    @assert dd.pars[:α] == 0 && dd.pars[:τ] <= minimum(dd.gr[:y])

    save("dd_comp.jld2", "dd", dd)
    mpe_simul!(dd, simul=show_simul)

    Ny = length(dd.gr[:y])
    v_noncont = dd.v[:V][1, ceil(Int, Ny / 2)]

    if DEP
        _, DEP_noncont = simul_dist(dd, K=1_000, burn_in=1_000, T=240)
        print("DEP with noncontingent: $DEP_noncont\n")
    end

    dd.pars[:α] = 1
    dd.pars[:min_q] = 0

    mpe_simul!(dd, maxiter=500, K=8, initialrep=false, simul=false)

    v_linear = dd.v[:V][1, ceil(Int, Ny / 2)]

    if DEP
        _, DEP_linear = simul_dist(dd, K=1_000, burn_in=1_000, T=240)
        print("DEP with linear: $DEP_linear\n")
    end


    dd.pars[:τ] = 1

    mpe_simul!(dd, maxiter=500, K=8, initialrep=false, simul=false)

    v_threshold = dd.v[:V][1, ceil(Int, Ny / 2)]
    if DEP
        _, DEP_threshold = simul_dist(dd, K=1_000, burn_in=1_000, T=240)
        print("DEP with threshold: $DEP_threshold\n")
    end

    dd = load("dd_comp.jld2", "dd")

    return v_noncont, v_linear, v_threshold
end

function comp_t5(dd::DebtMod)
    @assert dd.pars[:θ] >= 1e-3

    print("Solving original model with θ = $(dd.pars[:θ])\n")
    save("dd_comp_theta.jld2", "dd", dd)
    rob_n, rob_l, rob_t = comp_argbond(dd, show_simul=true, DEP = true)

    dd.pars[:θ] = 0
    print("Solving same model with rational expectations\n")
    rat_n, rat_l, rat_t = comp_argbond(dd, show_simul=true)

    dd = load("dd_comp_theta.jld2", "dd")

    return rob_n, rob_l, rob_t, rat_n, rat_l, rat_t
end
