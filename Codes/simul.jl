struct SimulPath
    names::Dict{Symbol,Int64}
    data::Matrix{Float64}
end

function SimulPath(T, nv::Vector{Symbol})
    K = length(nv)
    data = Matrix{Float64}(undef, T, K)
    names = Dict(key => jk for (jk, key) in enumerate(nv))

    return SimulPath(names, data)
end

horizon(pp::SimulPath) = size(pp.data, 1)

Base.getindex(pp::SimulPath, s::Symbol) = pp.data[:, pp.names[s]]

Base.getindex(pp::SimulPath, j::Int64, s::Symbol) = pp.data[j, pp.names[s]]
Base.getindex(pp::SimulPath, s::Symbol, j::Int64) = pp.data[j, pp.names[s]]

Base.setindex!(pp::SimulPath, x::Real, j::Int64, s::Symbol) = (pp.data[j, pp.names[s]] = x)
Base.setindex!(pp::SimulPath, x::Real, s::Symbol, j::Int64) = (pp.data[j, pp.names[s]] = x)

subpath(pp::SimulPath, t0, t1) = SimulPath(pp.names, pp.data[t0:t1, :])

function stationary_distribution(dd::DebtMod)
    Ny, ρy, σy, std_devs = (dd.pars[key] for key in (:Ny, :ρy, :σy, :std_devs))

    mc = tauchen(Int(Ny), ρy, σy, 0, Int(std_devs))

    stationary_distributions(mc)[1]
end


function iter_simul(b0, y0, def::Bool, ϵv, ξv, itp_R, itp_D, itp_prob, itp_c, itp_b, itp_q, itp_qD, itp_yield, pars::Dict, min_y, max_y, itp_qRE, itp_qdRE, itp_spr_og)
    ρy, σy, ψ, ℏ, r = (pars[sym] for sym in (:ρy, :σy, :ψ, :ℏ, :r))

    if def
        ct = itp_c(b0, y0, 2)
        bp = b0
        q = itp_qD(bp, y0)
        qRE = itp_qdRE(bp, y0)
        yield_MTI = itp_spr_og(bp, y0, 2)
    else
        ct = itp_c(b0, y0, 1)
        bp = itp_b(b0, y0)
        q = itp_q(bp, y0)
        qRE = itp_qRE(bp, y0)
        yield_MTI = itp_spr_og(bp, y0, 1)        
    end

    vD = itp_D(b0, y0)
    vR = itp_R(b0, y0)
    
    net_vR = vR - vD

    yield = itp_yield(q, y0)

    yield_RE = itp_yield(qRE, y0)
    
    spread_RE = yield_RE - r    
    spread_MTI = yield_MTI - r
    spread = yield - r

    # ϵ = rand(Normal(0, 1))
    yp = exp(ρy * log(y0) + σy * ϵv)

    yp = min(max_y, max(min_y, yp))

    if def
        prob_def = 1 - ψ
    else
        prob_def = itp_prob(bp, yp)
    end

    # ξ_def = rand()
    def_p = (ξv <= prob_def)

    new_def = (!def && def_p)

    if new_def
        bp = (1 - ℏ) * bp
    end

    return net_vR, def_p, ct, bp, yp, new_def, q, spread, spread_RE, spread_MTI
end

function simulvec(dd::DebtMod, itp_yield, itp_qRE, itp_qdRE, itp_spr_og, ϵvv, ξvv; burn_in=200, cond_defs = 35, separation = 4, stopdef = true, B0 = 0.0, Y0 = mean(dd.gr[:y]))

    # Length of simulation
    cd_sep = cond_defs + separation

    # Make sure shock vectors are consistent
    @assert length(ϵvv) == length(ξvv)
    K = length(ϵvv)

    # Prep/preallocate vector of simulation paths
    pv = Vector{SimulPath}(undef, K)
    
    # Set up interpolators
    knots    = (dd.gr[:b], dd.gr[:y])
    itp_R    = interpolate(knots, dd.v[:R], Gridded(Linear()))
    itp_D    = interpolate(knots, dd.v[:D], Gridded(Linear()))
    itp_v    = interpolate(knots, dd.v[:V], Gridded(Linear()))
    itp_prob = interpolate(knots, dd.v[:prob], Gridded(Linear()))
    itp_b    = interpolate(knots, dd.gb, Gridded(Linear()))

    itp_q    = interpolate(knots, dd.q, Gridded(Linear()))
    itp_qD   = interpolate(knots, dd.qD, Gridded(Linear()))

    knots    = (dd.gr[:b], dd.gr[:y], 1:2)
    itp_c    = interpolate(knots, dd.gc, Gridded(Linear()))

    Threads.@threads for jp in eachindex(pv)
        # Starting point
        b0 = B0
        y0 = Y0
        def = false
        new_def = false

        min_y, max_y = extrema(dd.gr[:y])

        # Vector of shocks for simulation jp is jp'th element of shock-vector vector
        ϵvec = ϵvv[jp]
        ξvec = ξvv[jp]

        Tmax = length(ϵvec)
        pp = SimulPath(Tmax, [:c, :b, :y, :v, :ζ, :fund_def, :def, :q, :spread, :sp_RE, :sp_MTI, :sp, :acc])

        contflag = true

        t = 0
        while contflag && t < Tmax
            t += 1

            # Time-t state known from before
            ϵv = ϵvec[t]
            ξv = ξvec[t]

            pp[:y, t] = y0
            pp[:b, t] = b0

            pp[:ζ, t] = ifelse(def, 2, 1)
            pp[:def, t] = ifelse(new_def, 1, 0)
            pp[:acc, t] = 0
            if new_def || !def
                pp[:acc, t] = 1
            end

            pp[:v, t] = itp_v(b0, y0)

            # Compute endogenous variables at time t
            net_vR, def, ct, b0, y0, new_def, q, spread, spread_RE, spread_MTI = iter_simul(b0, y0, def, ϵv, ξv, itp_R, itp_D, itp_prob, itp_c, itp_b, itp_q, itp_qD, itp_yield, dd.pars, min_y, max_y, itp_qRE, itp_qdRE, itp_spr_og)

            # Time-t stuff that is a function of the state
            pp[:fund_def, t] = ifelse(new_def && net_vR > 0, 1.0, 0.0)

            pp[:q, t] = q
            pp[:spread, t] = (1+spread)^4 - 1 ## annualized
            pp[:sp_RE, t] = (1+spread_RE)^4 - 1
            pp[:sp_MTI, t] = (1+spread_MTI)^4 - 1

            pp[:sp, t] = (1+get_spread(q, dd))^4 - 1
            pp[:c, t] = ct

            # Check stopping condition
            if stopdef && t >= burn_in + cd_sep && pp[:ζ, t] == 2 && maximum(pp[:ζ, jj] for jj in t-cd_sep+1:t-1) == 1
                contflag = false
            end
        end

        # Trim shock vector to final length of simulation
        ϵvv[jp] = ϵvv[jp][1:t]
        ξvv[jp] = ξvv[jp][1:t]
        
        # Trim burn-in period and/or unneeded history
        if stopdef
            pv[jp] = subpath(pp, t - cond_defs + 1, t)
        else
            pv[jp] = subpath(pp, 1+burn_in, Tmax)
        end
    end
    return pv, ϵvv, ξvv
end

compute_defprob(pv) = 100 * mean( 1-(1-sum(pp[:def])/sum(pp[:acc]))^4 for pp in pv )

function compute_moments(pv::Vector{SimulPath})

    moments = Dict{Symbol,Float64}()

    ## measure in basis points, already annualized
    moments[:mean_spr] = mean(mean(pp[:spread]) * 1e4 for pp in pv)

    moments[:sp_RE] = mean(mean(pp[:sp_RE]) * 1e4 for pp in pv)
    moments[:sp_MTI] = mean(mean(pp[:sp_MTI]) * 1e4 for pp in pv)

    moments[:std_spr]  = mean(std(pp[:spread].*1e4) for pp in pv)

    # Compare debt to annual GDP
    moments[:debt_gdp] = mean(mean(pp[:b]./pp[:y]) * 25 for pp in pv)

    moments[:rel_vol]  = mean(std(log.(pp[:c])) ./ std(log.(pp[:y])) for pp in pv)

    moments[:corr_yc]  = mean(cor(pp[:c], pp[:y]) for pp in pv)

    moments[:corr_ytb] = mean(cor(pp[:y], pp[:y] .- pp[:c]) for pp in pv)

    moments[:corr_ysp] = mean(cor(pp[:y], pp[:spread]) for pp in pv)

    return moments
end

get_targets() = targets_CE()

targets_CE() = Dict{Symbol, Float64}(
    :mean_spr => 815,
    :std_spr => 443,
    :debt_gdp => 17.4,
    :def_prob => 3,
    # :def_prob => 5.4,
    :rel_vol => 0.87,
    :corr_yc => 0.97,
    :corr_ytb => -0.77,
    :corr_ysp => -0.72,
    :sp_RE => NaN,
    :sp_MTI => NaN,
    :DEP => NaN,
)

function table_during(pv::Vector{SimulPath}, pv_uncond::Vector{SimulPath})

    syms = [:mean_spr, :std_spr, :debt_gdp, :def_prob]

    targets = get_targets()

    moments = compute_moments(pv)
    moments[:def_prob] = compute_defprob(pv_uncond)

    names = ["Spread", "Std Spread", "Debt-to-GDP", "Default Prob"]
    maxn = maximum(length(name) for name in names)

    table = "\n"
    table *= ("$(rpad("", maxn+3, " "))")
    table *= ("$(rpad("Data", 10, " "))")
    table *= ("$(rpad("Model", 10, " "))")
    table *= ("$(rpad("Contrib.", 10, " "))")
    table *= "\n"

    for (jn, nv) in enumerate(names)
    
        table *= ("$(rpad(nv, maxn+3, " "))")
        table *= ("$(rpad(@sprintf("%0.3g", targets[syms[jn]]), 10, " "))")
        table *= ("$(rpad(@sprintf("%0.3g", moments[syms[jn]]), 10, " "))")
        contr = 100 * (moments[syms[jn]] / targets[syms[jn]] - 1)^2
        table *= ("$(rpad(@sprintf("%0.3g", contr), 10, " "))")
        table *= ("\n")
    end

    print(table)
    nothing
end

function table_moments(pv::Vector{SimulPath}, pv_uncond::Vector{SimulPath}, pv_RE=[], pv_uncond_RE=[]; savetable=false)

    syms = [:mean_spr,
    # :mean_sp,
    # :sp_RE, :sp_MTI,
    :std_spr, :debt_gdp, :def_prob, :rel_vol, :corr_yc, :corr_ytb, :corr_ysp]

    targets = get_targets()

    moments = compute_moments(pv)
    # Number of defaults divided total periods with market access (ζ = 1)
    moments[:def_prob] = compute_defprob(pv_uncond)

    # moments[:mean_spr] = mean(mean(pp[:spread]) for pp in pv_uncond) * 1e4

    if length(pv_RE) > 0
        moments_RE = compute_moments(pv_RE)
        moments_RE[:def_prob] = compute_defprob(pv_uncond_RE)
        # moments[:mean_spr] = mean(mean(pp[:spread]) for pp in pv_uncond_RE) * 1e4
    end

    names = ["Spread",
    # "Spread OG",
    # "o/w Spread RE", "Spread MTI",
    "Std Spread", "Debt-to-GDP", "Default Prob", "Std(c)/Std(y)", "Corr(y,c)", "Corr(y,tb/y)", "Corr(y,spread)"]
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

        table *= ("$(rpad(nv, maxn+2, " "))")
        table *= ("& $(rpad(@sprintf("%0.3g", targets[syms[jn]]), 10, " "))")
        table *= ("& $(rpad(@sprintf("%0.3g", moments[syms[jn]]), 10, " "))")
        if length(pv_RE) > 0
            table *= ("& $(rpad(@sprintf("%0.3g", moments_RE[syms[jn]]), 10, " "))")
        end
        table *= ("\\\\\n")
    end
    table *= "\\bottomrule"
    print(title)
    print(table)
    savetable && write("table_targets.txt", table)
    nothing
end

function simulshocks(T, K)
    Random.seed!(25)
    ϵvv = [rand(Normal(0,1), T) for _ in 1:K]
    ξvv = [rand(T) for _ in 1:K]

    return ϵvv, ξvv
end

function eval_gmm(pv, pv_uncond, savetable, showtable, smalltable)
    targets = get_targets()
    keys = [:mean_spr, :std_spr, :debt_gdp, :def_prob]

    moments = compute_moments(pv)
    moments[:def_prob] = compute_defprob(pv_uncond)

    targets_vec = [targets[key] for key in keys]
    moments_vec = [moments[key] for key in keys]

    W = diagm(ones(4))

    showtable && table_moments(pv, pv_uncond, savetable = savetable)
    !showtable && smalltable && table_during(pv, pv_uncond)

    objective = (moments_vec ./ targets_vec .- 1)' * W * (moments_vec ./ targets_vec .- 1)
    objective, targets_vec, moments_vec
end


function calib_targets(dd::DebtMod; cond_T=2000, cond_K=1_000, kwargs...)

    ϵvv, ξvv = simulshocks(cond_T, cond_K);

    objective, targets_vec, moments_vec = calib_targets(dd, ϵvv, ξvv; kwargs...)
    
    objective, targets_vec, moments_vec, ϵvv, ξvv
end

function calib_targets(dd::DebtMod, ϵvv, ξvv; uncond_K=2_000, uncond_burn=2_000, uncond_T=4_000, savetable=false, showtable=(savetable || false), smalltable=false, do_calc = showtable)

    ϵvv_unc, ξvv_unc = simulshocks(uncond_T, uncond_K);

    itp_yield = get_yields_itp(dd);
    showtable && print("yields ✓ ")
    itp_qRE, itp_qdRE = q_RE(dd, do_calc = do_calc);
    showtable && print("RE ✓ ")
    itp_spr_og = itp_mti(dd, do_calc = do_calc);
    showtable && print("MTI ✓\n")

    pv_uncond, _ = simulvec(dd, itp_yield, itp_qRE, itp_qdRE, itp_spr_og, ϵvv_unc, ξvv_unc, burn_in=uncond_burn, stopdef=false)

    pv, ϵvv, ξvv = simulvec(dd, itp_yield, itp_qRE, itp_qdRE, itp_spr_og, ϵvv, ξvv)

    objective, targets_vec, moments_vec = eval_gmm(pv, pv_uncond, savetable, showtable, smalltable)

    objective, targets_vec, moments_vec
end

function simul_table(dd::DebtMod, dd_RE::DebtMod; cond_K=1_000, uncond_K=10_000, uncond_burn=2_000, uncond_T=4_000, cond_T=2000, longrun = false, kwargs...)

    ϵvv_unc, ξvv_unc = simulshocks(uncond_T, uncond_K);
    ϵvv, ξvv = simulshocks(cond_T, cond_K);


    itp_yield = get_yields_itp(dd);
    itp_qRE, itp_qdRE = q_RE(dd, do_calc = true);
    itp_spr_og = itp_mti(dd, do_calc = false);


    pv_uncond, _ = simulvec(dd, itp_yield, itp_qRE, itp_qdRE, itp_spr_og, ϵvv_unc, ξvv_unc, burn_in=uncond_burn, stopdef=false)
    pv, _ = simulvec(dd, itp_yield, itp_qRE, itp_qdRE, itp_spr_og, ϵvv, ξvv)
    
    
    itp_yield_RE = get_yields_itp(dd_RE)
    itp_qRE, itp_qdRE = q_RE(dd_RE, do_calc = true);
    itp_spr_og = itp_mti(dd, do_calc = false);

    pv_uncond_RE, _ = simulvec(dd_RE, itp_yield_RE, itp_qRE, itp_qdRE, itp_spr_og, ϵvv_unc, ξvv_unc, burn_in=uncond_burn, stopdef=false)
    pv_RE, _ = simulvec(dd_RE, itp_yield_RE, itp_qRE, itp_qdRE, itp_spr_og, ϵvv, ξvv)

    if longrun
        return table_moments(pv_uncond, pv_uncond, pv_uncond_RE, pv_uncond_RE; kwargs...)
    else
        table_moments(pv, pv_uncond, pv_RE, pv_uncond_RE; kwargs...)
    end
end


function update_dd!(dd::DebtMod, params::Dict)
    for (key, val) in params
        if haskey(dd.pars, key)
            dd.pars[key] = val
        end
    end
end

function calibrate(dd::DebtMod, targets=get_targets(); factor=0.1,
    minβ=1 / (1 + 0.15),
    mind1=-0.5,
    mind2=0.2,
    minθ=0.005,
    maxβ=1 / (1 + 0.015),
    maxd1=-0.01,
    maxd2=0.4,
    maxθ=2.5
)

    maxβ = min(0.99, maxβ)
    minθ = max(1e-3, minθ)

    keys = [:mean_spr, :std_spr, :debt_gdp, :rel_vol, :corr_yc, :corr_ytb, :corr_ysp, :def_prob]

    function objective(x)
        β = x[1]
        d1 = x[2]
        d2 = x[3]
        θ = x[4]

        dd.pars[:β] = β
        dd.pars[:d1] = d1
        dd.pars[:d2] = d2
        dd.pars[:θ] = θ

        print("Trying with (β, d1, d2, θ) = ($(@sprintf("%0.4g", dd.pars[:β])), $(@sprintf("%0.4g", dd.pars[:d1])), $(@sprintf("%0.4g", dd.pars[:d2])), $(@sprintf("%0.4g", dd.pars[:θ]))): ")

        mpe!(dd, min_iter=25, maxiter = 1_250, tol=1e-6, tinyreport=true)

        w, t, m, _, _ = calib_targets(dd, smalltable=false, cond_K=7_500, uncond_K=10_000)
        print("v = $(@sprintf("%0.3g", 100*w))\n")
        return w
    end

    xmin = [minβ, mind1, mind2, minθ]
    xmax = [maxβ, maxd1, maxd2, maxθ]
    xguess = [dd.pars[key] for key in [:β, :d1, :d2, :θ]]

    # res = Optim.optimize(objective, xmin, xmax, xguess, Fminbox(NelderMead())) # se traba en mínimos locales
    # Simulated Annealing no tiene Fminbox 
    res = Optim.optimize(objective, xguess, ParticleSwarm(lower=xmin, upper=xmax, n_particles=3))
end

calib_range(dd::DebtMod; rb, r1, r2, rt) = calibrate(dd;
    minβ = dd.pars[:β]  - rb,
    maxβ = dd.pars[:β]  + rb,
    mind1= dd.pars[:d1] - r1,
    maxd1= dd.pars[:d1] + r1,
    mind2= dd.pars[:d2] - r2,
    maxd2= dd.pars[:d2] + r2,
    minθ = dd.pars[:θ]  - rt,
    maxθ = dd.pars[:θ]  + rt,
)
calib_close(dd::DebtMod; factor = 0.1) = calibrate(dd; factor = 0.1,
    minβ = dd.pars[:β] * (1 - factor),
    maxβ = dd.pars[:β] * (1 + factor),
    mind1= dd.pars[:d1]* (1 + factor), # d1 is negative in most of them
    maxd1= dd.pars[:d1]* (1 - factor),
    mind2= dd.pars[:d2]* (1 - factor),
    maxd2= dd.pars[:d2]* (1 + factor),
    minθ = dd.pars[:θ] * (1 - factor),
    maxθ = dd.pars[:θ] * (1 + factor),
)

function getval(key, pars::Dict)

    if haskey(pars, key)
        v = pars[key]
    elseif key == :ρ
        v = pars[:β]^-1 - 1
    elseif key == :λ0
        v = pars[:d2] + pars[:d1]
    elseif key == :λ1
        v = -pars[:d1]
    end

    return v
end

function setval!(pars::Dict, key, val)

    if haskey(pars, key)
        pars[key] = val
    elseif key == :ρ
        pars[:β] = (1+val)^-1
    elseif key == :λ0
        λ1 = getval(:λ1, pars)
        pars[:d2] = val + λ1
    elseif key == :λ1
        pars[:d1] = -val
    end
end

function eval_Sobol(dd::DebtMod, key, val, tol, verbose)
    setval!(dd.pars, key, val)

    w = mpe_simul!(dd, tol = tol, simultable = verbose, initialrep = false)
    verbose || print("w=$(@sprintf("%0.3g", w)) ")
    w
end

function iter_Sobol(dd::DebtMod, key, σ, tol; Nx = 15)
    
    x = getval(key, dd.pars)
    xvec = range(x-σ, x+σ, length = Nx)
    jc = ceil(Int, Nx/2)

    W = Inf
    xopt = x
    jopt = jc
    
    for (jx, xv) in enumerate(xvec)

        w = eval_Sobol(dd, key, xv, tol, (jx==jc))
        jx == jc && print("w = $(@sprintf("%0.3g", w))\n")

        if w < W
            W = w
            xopt = xv
            jopt = jx
        end
    end

    return xopt, jopt, W, x
end


function pseudoSobol!(dd::DebtMod, best_p = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ));
    maxiter = 500, tol = 1e-6, 
    σβ = 0.0005, σθ = 0.01, σ1 = 0.0005, σ2 = 0.0005)

    update_dd!(dd, best_p)

    σvec = [σβ, σθ, σ1, σ2]
    names = [:β, :θ, :d1, :d2]
    Nxs = [5,5,5,5]

    curr_p = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ))
    W = Inf

    iter = 0
    while iter < maxiter
        js = 1 + (iter % length(names))
        
        iter += 1

        key = names[js]
        σ = σvec[js]/2

        Nx = Nxs[js]

        print("Iteration $iter at $(Dates.format(now(), "HH:MM")). Moving $key from $(curr_p)\n")
        xopt, jopt, w, x_og = iter_Sobol(dd, key, σ, tol, Nx=Nx)

        print("\nBest objective: $(@sprintf("%0.3g", w)) at $key [$jopt] = $(@sprintf("%0.5g", xopt)) in [$(@sprintf("%0.5g", x_og-σ)), $(@sprintf("%0.5g", x_og+σ))]. ")

        setval!(curr_p, key, xopt)
        mpe!(dd, min_iter = 25, maxiter = 1_000, tol = tol, tinyreport = true)

        if w < W
            W = w
            for (key, val) in curr_p
                best_p[key] = val
            end
        end
        print("Best so far $(@sprintf("%0.3g", W))\n")

        update_dd!(dd, curr_p)
    end
end

function mpe_simul!(dd::DebtMod; K = 3, min_iter = 25, maxiter = 600, tol = 1e-6, simul = true, cond_K = 7_500, uncond_K = 10_000, initialrep = simul, simultable=true)

    initialrep && print("Solving with (β, d1, d2, θ) = ($(@sprintf("%0.4g", dd.pars[:β])), $(@sprintf("%0.4g", dd.pars[:d1])), $(@sprintf("%0.4g", dd.pars[:d2])), $(@sprintf("%0.4g", dd.pars[:θ])))\n")

    for _ in 1:K
        mpe!(dd, min_iter = min_iter, maxiter = maxiter, tol = tol, tinyreport = true)
    end

    initialrep && mpe!(dd, min_iter = min_iter, maxiter = maxiter, tol = tol, verbose = false)

    if simul
        w,t,m,_, _ = calib_targets(dd, cond_K = cond_K, uncond_K = uncond_K, smalltable=simultable);
        return 100w
    end
end

# pmat, DEP = simul_dist(dd, K = 1_000, burn_in = 1_000, T = 240); DEP