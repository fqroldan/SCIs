function distorted_density(dd::DebtMod)
    Nb, Ny = (length(dd.gr[key]) for key in (:b, :y))
    θ, ρy, σy, ψ = (dd.pars[key] for key in (:θ, :ρy, :σy, :ψ))

    M = zeros(Nb, Ny, 2, Ny)
    orig_dens = zeros(Nb, Ny, 2, Ny)

    for (jy, yv) in enumerate(dd.gr[:y])
        Ey = ρy * log(yv)
        for (jbp, bpv) in enumerate(dd.gr[:b]), jζ in 1:2
            
            Em = 0.0
            sum_prob = 0.0            
            for (jyp, ypv) in enumerate(dd.gr[:y])
                prob = pdf(Normal(Ey, σy), ypv)
                sum_prob += prob
                
                if jζ == 1 # Repayment
                    prob_def = dd.v[:prob][jbp, jyp]
                else
                    prob_def = 1 - ψ
                end

                kernel = prob_def * exp(-θ * dd.vL[jbp, jyp, 2]) + (1 - prob_def) * exp(-θ * dd.vL[jbp, jyp, 1])

                orig_dens[jbp, jy, jζ, jyp] = prob

                
                M[jbp, jy, jζ, jyp] = kernel
                Em += kernel * prob
            end

            M[jbp, jy, jζ, :] ./= Em
        end
    end
    dist_dens = M .* orig_dens

    return M, orig_dens, dist_dens
end

function dist_inv_cdf(dd::DebtMod, K=50)
    Nb, Ny = (length(dd.gr[key]) for key in (:b, :y))

    min_y, max_y = extrema(dd.gr[:y])
    yvec = range(min_y, max_y, length=K)

    M, orig_dens, dist_dens = distorted_density(dd)

    int_dens = zeros(Nb, Ny, 2, K)
    for jbp in eachindex(dd.gr[:b]), jy in eachindex(dd.gr[:y]), jζ in 1:2

        itp_dist_dens = interpolate((dd.gr[:y],), dist_dens[jbp, jy, jζ, :], Gridded(Linear()))

        f(y′) = itp_dist_dens(y′)

        total_cdf = hquadrature(f, min_y, max_y)[1]

        int_dens[jbp, jy, jζ, :] .= [hquadrature(f, min_y, y′)[1] for y′ in yvec] / total_cdf

    end

    int_itp = interpolate((dd.gr[:b], dd.gr[:y], 1:2, yvec), int_dens, Gridded(Linear()))
end

### HERE

function distorted_transitions(dd::DebtMod)
    θ, ψ = (dd.pars[key] for key in (:θ, :ψ))

    Nb = length(dd.gr[:b])
    Ny = length(dd.gr[:y])
    Nζ = 2

    knots = (dd.gr[:b], 1:Ny, 1:2)
    itp_vL = interpolate(knots, dd.vL, (Gridded(Linear()), NoInterp(), NoInterp()))

    dist_P = Array{Float64,4}(undef, Nb, Ny, Nζ, Ny)

    for (jbp, bpv) in enumerate(dd.gr[:b]), jy in eachindex(dd.gr[:y]), jζ in 1:2

        sum_prob = 0.0
        for jyp in eachindex(dd.gr[:y])

            if jζ == 1 # Repayment
                prob_def = dd.v[:prob][jbp, jyp]
                cond_sdf = prob_def * exp(-θ * itp_vL(bpv, jyp, 2)) + (1 - prob_def) * exp(-θ * dd.vL[jbp, jyp, 1])
            else
                prob_def = (1 - ψ) + ψ * dd.v[:prob][jbp, jyp]
                cond_sdf = prob_def * exp(-θ * dd.vL[jbp, jyp, 2]) + (1 - prob_def) * exp(-θ * dd.vL[jbp, jyp, 1])
            end

            prob = dd.P[:y][jy, jyp] * cond_sdf

            sum_prob += prob
            dist_P[jbp, jy, jζ, jyp] = prob
        end
        dist_P[jbp, jy, jζ, :] ./= sum_prob
    end

    return dist_P
end

# Simul with dist d, evaluate likelihood for both d and d'

function iter_simul_dist(b0, y0, def::Bool, itp_P_true, itp_P_alt, itp_prob, itp_b, pars::Dict, ygrid)
    ψ, ℏ = (pars[sym] for sym in (:ψ, :ℏ))

    if def
        bp = b0
        jζ = 2
    else
        bp = itp_b(b0, y0)
        jζ = 1
    end

    probs = Weights([itp_P_true(bp, y0, jζ, yv) for yv in ygrid])
    yp = sample(ygrid, probs)

    # probs = cumsum(itp_P_true(bp, y0, jζ, yv) for yv in ygrid)
    # jyp = findfirst(rand() .<= probs)
    # yp = ygrid[jyp]
    
    loglik_true = log(itp_P_true(bp, y0, jζ, yp))
    loglik_alt  = log(itp_P_alt(bp, y0, jζ, yp))

    if def
        prob_def = 1 - ψ
    else
        prob_def = itp_prob(bp, yp)
    end

    ξ_def = rand()
    def_p = (ξ_def <= prob_def)

    new_def = (!def && def_p)

    if new_def
        bp = (1 - ℏ) * bp
    end

    return def_p, bp, yp, loglik_true, loglik_alt
end

function simul_eval(itp_P_true, itp_P_alt, itp_prob, itp_b, pars, ygrid, burn_in, T)

    b0 = 0.0
    y0 = mean(ygrid)
    def_0 = false

    pp = SimulPath(T, [:ζ, :b, :y, :loglik_true, :loglik_alt])
    loglik_true = 0.0
    loglik_alt = 0.0

    for t in 1:burn_in+T

        tt = t - burn_in
        if tt > 0
            pp[:y, tt] = y0
            pp[:b, tt] = b0
            pp[:ζ, tt] = ifelse(def_0, 2, 1)
        end

        def_0, b0, y0, loglik_true_t, loglik_alt_t = iter_simul_dist(b0, y0, def_0, itp_P_true, itp_P_alt, itp_prob, itp_b, pars, ygrid)

        if tt > 0
            loglik_true += loglik_true_t
            loglik_alt += loglik_alt_t
        end
    end

    prob_wrong = (loglik_true < loglik_alt) + 0.5 * (loglik_true == loglik_alt)
    return pp, prob_wrong
end

# K = 2_000, burn_in = 2_000, T = 240
function simul_dist(dd::DebtMod; K = 1_000, burn_in = 800, T = 240)
    ygrid = range(extrema(dd.gr[:y])..., length=21)
    pars = dd.pars

    Random.seed!(25)

    dist_P = distorted_transitions(dd)

    orig_P = [dd.P[:y][jy, jyp] for b in dd.gr[:b], jy in eachindex(dd.gr[:y]), jζ in 1:2, jyp in eachindex(dd.gr[:y])]

    knots = (dd.gr[:b], dd.gr[:y], 1:2, dd.gr[:y])
    itp_P_dist = interpolate(knots, dist_P, Gridded(Linear()))
    itp_P_orig = interpolate(knots, orig_P, Gridded(Linear()))

    knots = (dd.gr[:b], dd.gr[:y])
    itp_b = interpolate(knots, dd.gb, Gridded(Linear()))
    itp_prob = interpolate(knots, dd.v[:prob], Gridded(Linear()))
    
    pmat = Matrix{SimulPath}(undef, K, 2)
    
    prob_wrong = 0.0
    Threads.@threads for k in 1:K

        pp_from_orig, wrong_from_orig = simul_eval(itp_P_orig, itp_P_dist, itp_prob, itp_b, pars, ygrid, burn_in, T)
        pp_from_dist, wrong_from_dist = simul_eval(itp_P_dist, itp_P_orig, itp_prob, itp_b, pars, ygrid, burn_in, T)

        prob_wrong += (wrong_from_orig + wrong_from_dist) / 2

        pmat[k, 1] = pp_from_orig
        pmat[k, 2] = pp_from_dist
    end

    prob_wrong /= K

    return pmat, prob_wrong
end

    

