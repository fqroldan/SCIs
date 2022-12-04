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

function iter_simul_dist(b0, y0, def::Bool, itp_P_true, itp_P_alt, itp_prob, itp_b, pars::Dict, ygrid, ξt, ϵt)
    ψ, ℏ = (pars[sym] for sym in (:ψ, :ℏ))

    if def
        bp = b0
        jζ = 2
    else
        bp = itp_b(b0, y0)
        jζ = 1
    end

    # probs = Weights([itp_P_true(bp, y0, jζ, yv) for yv in ygrid])
    # yp = sample(rng, ygrid, probs)
    weights = cumsum(itp_P_true(bp, y0, jζ, yv) for yv in ygrid)
    weights *= 1/weights[end]
    jyp = min(searchsortedfirst(weights, ϵt), length(ygrid))
    yp = ygrid[jyp]
    
    loglik_true = log(itp_P_true(bp, y0, jζ, yp))
    loglik_alt  = log(itp_P_alt(bp, y0, jζ, yp))

    if def
        prob_def = 1 - ψ
    else
        prob_def = itp_prob(bp, yp)
    end

    ξ_def = ξt 
    def_p = (ξ_def <= prob_def)

    new_def = (!def && def_p)

    if new_def
        bp = (1 - ℏ) * bp
    end

    return def_p, bp, yp, loglik_true, loglik_alt
end

function simul_eval(itp_P_true, itp_P_alt, itp_prob, itp_b, pars, ygrid, burn_in, T, ξvec, ϵvec)

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
        
        ξt = ξvec[t]
        ϵt = ϵvec[t]

        def_0, b0, y0, loglik_true_t, loglik_alt_t = iter_simul_dist(b0, y0, def_0, itp_P_true, itp_P_alt, itp_prob, itp_b, pars, ygrid, ξt, ϵt)

        if tt > 0
            loglik_true += loglik_true_t
            loglik_alt += loglik_alt_t
        end
    end

    prob_wrong = (loglik_true < loglik_alt) + 0.5 * (loglik_true == loglik_alt)
    return pp, prob_wrong
end

function simul_dist(dd::DebtMod; K = 2_000, burn_in = 2_000, T = 35)
    ygrid = range(extrema(dd.gr[:y])..., length=21)
    pars = dd.pars

    rng = MersenneTwister(1989)

    ϵvv = [rand(rng, burn_in + T) for _ in 1:K]
    ξvv = [rand(rng, burn_in + T) for _ in 1:K]

    dist_P = distorted_transitions(dd)

    orig_P = [dd.P[:y][jy, jyp] for b in dd.gr[:b], jy in eachindex(dd.gr[:y]), jζ in 1:2, jyp in eachindex(dd.gr[:y])]

    knots = (dd.gr[:b], dd.gr[:y], 1:2, dd.gr[:y])
    itp_P_dist = interpolate(knots, dist_P, Gridded(Linear()))
    itp_P_orig = interpolate(knots, orig_P, Gridded(Linear()))

    knots = (dd.gr[:b], dd.gr[:y])
    itp_b = interpolate(knots, dd.gb, Gridded(Linear()))
    itp_prob = interpolate(knots, dd.v[:prob], Gridded(Linear()))
    
    pmat = Matrix{SimulPath}(undef, K, 2)
    wrong_vec = Vector{Float64}(undef, K)
    
    Threads.@threads for k in 1:K

        ϵvec = ϵvv[k]
        ξvec = ξvv[k]

        pp_from_orig, wrong_from_orig = simul_eval(itp_P_orig, itp_P_dist, itp_prob, itp_b, pars, ygrid, burn_in, T, ξvec, ϵvec)
        pp_from_dist, wrong_from_dist = simul_eval(itp_P_dist, itp_P_orig, itp_prob, itp_b, pars, ygrid, burn_in, T, ξvec, ϵvec)

        wrong_vec[k] = (wrong_from_orig + wrong_from_dist) / 2

        pmat[k, 1] = pp_from_orig
        pmat[k, 2] = pp_from_dist
    end

    prob_wrong = mean(wrong_vec)

    return pmat, prob_wrong
end

    

