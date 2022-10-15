function comp_argbond(dd::DebtMod; show_simul=false)
    @assert dd.pars[:α] == 0 && dd.pars[:τ] <= minimum(dd.gr[:y])

    save("dd_comp.jld2", "dd", dd)
    mpe_simul!(dd, simul=show_simul)

    Ny = length(dd.gr[:y])
    v_noncont = dd.v[:V][1, ceil(Int, Ny/2)]

    dd.pars[:α] = 1
    dd.pars[:min_q] = 0

    mpe_simul!(dd, maxiter = 500, K = 8, initialrep = false, simul=false)

    v_linear = dd.v[:V][1, ceil(Int, Ny/2)]
    
    dd.pars[:τ] = 1
    
    mpe_simul!(dd, maxiter = 500, K = 8, initialrep = false, simul=false)

    v_threshold = dd.v[:V][1, ceil(Int, Ny/2)]

    dd = load("dd_comp.jld2", "dd")

    return v_noncont, v_linear, v_threshold
end

