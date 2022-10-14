function comp_argbond(dd::DebtMod; show_simul=false)

    mpe_simul!(dd, K=2, simul=show_simul)

    Ny = length(dd.gr[:y])
    v_noncont = dd.v[:V][1, ceil(Int, Ny/2)]

    dd.pars[:α] = 1

    mpe_simul!(dd, K=2, initialrep = false, simul=false)

    v_linear = dd.v[:V][1, ceil(Int, Ny/2)]
    
    dd.pars[:τ] = 1
    dd.pars[:min_q] = 0
    
    mpe_simul!(dd, K=2, initialrep = false, simul=false)

    v_threshold = dd.v[:V][1, ceil(Int, Ny/2)]

    return v_noncont, v_linear, v_threshold
end

