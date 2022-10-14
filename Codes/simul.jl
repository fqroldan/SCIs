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


function iter_simul(b0, y0, def::Bool, ϵv, ξv, itp_R, itp_D, itp_prob, itp_c, itp_b, itp_q, itp_qD, itp_yield, pars::Dict, min_y, max_y)
    ρy, σy, ψ, ℏ, r = (pars[sym] for sym in (:ρy, :σy, :ψ, :ℏ, :r))

    if def
        v = itp_D(b0, y0)
        ct = itp_c(b0, y0, 2)
        bp = b0
        q = itp_qD(bp, y0)
    else
        v = itp_R(b0, y0)
        ct = itp_c(b0, y0, 1)
        bp = itp_b(b0, y0)
        q = itp_q(bp, y0)
    end

    spread = itp_yield(q, y0) - r
    
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

    return v, def_p, ct, bp, yp, new_def, q, spread
end

function simulvec(dd::DebtMod, K; burn_in=200, Tmax=10 * burn_in, cond_defs = 35, separation = 4, stopdef = true)

    cd_sep = cond_defs + separation
    
    ϵmat = rand(Normal(0,1), Tmax, K)
    ξmat = rand(Tmax, K)

    pv = Vector{SimulPath}(undef, K)

    itp_yield = get_yields_itp(dd)
    
    knots = (dd.gr[:b], dd.gr[:y])
    itp_R = interpolate(knots, dd.v[:R], Gridded(Linear()))
    itp_D = interpolate(knots, dd.v[:D], Gridded(Linear()))
    itp_v = interpolate(knots, dd.v[:V], Gridded(Linear()))
    itp_prob = interpolate(knots, dd.v[:prob], Gridded(Linear()))
    itp_b = interpolate(knots, dd.gb, Gridded(Linear()))

    itp_q = interpolate(knots, dd.q, Gridded(Linear()))
    itp_qD = interpolate(knots, dd.qD, Gridded(Linear()))

    knots = (dd.gr[:b], dd.gr[:y], 1:2)
    itp_c = interpolate(knots, dd.gc, Gridded(Linear()))

    Threads.@threads for jp in eachindex(pv)
        b0 = 0.0
        y0 = mean(dd.gr[:y])
        def = false
        new_def = false
    
        min_y, max_y = extrema(dd.gr[:y])

        ϵvec = ϵmat[:, jp]
        ξvec = ξmat[:, jp]
    
        pp = SimulPath(Tmax, [:c, :b, :y, :v, :ζ, :v_cond, :def, :q, :spread, :sp])
    
        contflag = true
    
        t = 0
        while contflag && t < Tmax
            t += 1

            ϵv = ϵvec[t]
            ξv = ξvec[t]
    
            pp[:y, t] = y0
            pp[:b, t] = b0
    
            pp[:ζ, t] = ifelse(def, 2, 1)
            pp[:def, t] = ifelse(new_def, 1, 0)
    
            pp[:v, t] = itp_v(b0, y0)
    
            v, def, ct, b0, y0, new_def, q, spread = iter_simul(b0, y0, def, ϵv, ξv, itp_R, itp_D, itp_prob, itp_c, itp_b, itp_q, itp_qD, itp_yield, dd.pars, min_y, max_y)
    
            pp[:v_cond, t] = v
    
            pp[:q, t] = q
            pp[:spread, t] = (1+spread)^4 - 1 ## annualized
            pp[:sp, t] = (1+get_spread(q, dd))^4 - 1
            pp[:c, t] = ct
    
            if stopdef && t >= burn_in + cd_sep && pp[:ζ, t] == 2 && maximum(pp[:ζ, jj] for jj in t-cd_sep+1:t-1) == 1
                contflag = false
            end
        end
        
        if stopdef
            pv[jp] = subpath(pp, t - cond_defs + 1, t)
        else
            pv[jp] = subpath(pp, 1+burn_in, Tmax)
        end
    end    
    return pv
end

function compute_moments(pv::Vector{SimulPath})

    moments = Dict{Symbol,Float64}()

    ## measure in basis points, already annualized
    moments[:mean_spr] = mean(mean(pp[:spread]) * 1e4 for pp in pv)
    moments[:mean_sp]  = mean(mean(pp[:sp]) * 1e4 for pp in pv)
    moments[:std_spr]  = mean(std(pp[:spread].*1e4) for pp in pv)

    # Compare debt to annual GDP
    moments[:debt_gdp] = mean(mean(pp[:b]./pp[:y]) * 25 for pp in pv)

    moments[:rel_vol]  = mean(std(log.(pp[:c])) ./ std(log.(pp[:y])) for pp in pv)

    moments[:corr_yc]  = mean(cor(pp[:c], pp[:y]) for pp in pv)

    moments[:corr_ytb] = mean(cor(pp[:y], pp[:y] .- pp[:c]) for pp in pv)

    moments[:corr_ysp] = mean(cor(pp[:y], pp[:spread]) for pp in pv)

    return moments
end

function PP_targets()
    targets = Dict{Symbol,Float64}(
        :mean_spr => 815,
        # :mean_spr => 740,
        :mean_sp  => 815,
        :std_spr => 458,
        # :std_spr => 250,
        # :debt_gdp => 25,
        :debt_gdp => 33,
        :rel_vol => 0.87,
        :corr_yc => 0.97,
        :corr_ytb => -0.77,
        :corr_ysp => -0.72,
        :def_prob => 5.4,
        # :def_prob => 3,
    )
end

function table_during(pv::Vector{SimulPath}, pv_uncond::Vector{SimulPath}, min_q)

    syms = [:mean_spr, :std_spr, :debt_gdp, :def_prob]

    targets = PP_targets()

    moments = compute_moments(pv)
    moments[:def_prob] = mean(sum(pp[:def]) / (sum(pp[:ζ].==1) / 4) * 100 for pp in pv_uncond)

    names = ["Spread", "Std Spread", "Debt-to-GDP", "Default Prob"]
    maxn = maximum(length(name) for name in names)

    freq_q = 100*mean(mean(p[:q] .<= min_q + 0.01) for p in pv)

    table = "\n"
    table *= ("$(rpad("", maxn+3, " "))")
    table *= ("$(rpad("Data", 10, " "))")
    table *= ("$(rpad("Bench.", 10, " "))")
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

    table *= "Freq. at min_q = $(@sprintf("%0.3g", freq_q))%\n"

    print(table)
    nothing
end

function table_moments(pv::Vector{SimulPath}, pv_uncond::Vector{SimulPath}, pv_RE=[], pv_uncond_RE=[]; savetable=false)

    syms = [:mean_spr,
    # :mean_sp,
    :std_spr, :debt_gdp, :rel_vol, :corr_yc, :corr_ytb, :corr_ysp, :def_prob]

    targets = PP_targets()

    moments = compute_moments(pv)
    # Number of defaults divided total periods with market access (ζ = 1)
    moments[:def_prob] = mean(sum(pp[:def]) / (sum(pp[:ζ].==1) / 4) * 100 for pp in pv_uncond)
    # moments[:mean_spr] = mean(mean(pp[:spread]) for pp in pv_uncond) * 1e4

    if length(pv_RE) > 0
        moments_RE = compute_moments(pv_RE)
        moments_RE[:def_prob] = mean(sum(pp[:def]) / (sum(pp[:ζ].==1) / 4) * 100 for pp in pv_uncond_RE)
        # moments[:mean_spr] = mean(mean(pp[:spread]) for pp in pv_uncond_RE) * 1e4
    end

    names = ["Spread",
    # "Spread OG",
    "Std Spread", "Debt", "Std(c)/Std(y)", "Corr(y,c)", "Corr(y,tb/y)", "Corr(y,spread)", "Default Prob"]
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

function simul_table(dd::DebtMod, K = 1_000; kwargs...)
    Random.seed!(25)

    pv_uncond = simulvec(dd, 2_000, burn_in=2_000, Tmax=4_000, stopdef=false)
    pv = simulvec(dd, K)

    table_moments(pv, pv_uncond; kwargs...)
end

function calib_targets(dd::DebtMod; cond_K = 1_000, uncond_K = 2_000 , uncond_burn = 2_000, uncond_T = 4_000, savetable=false, showtable=(savetable||false), smalltable=false)
    min_q = dd.pars[:min_q]
    
    targets = PP_targets()

    # keys = [:mean_spr,
    # # :mean_sp,
    # :std_spr, :debt_gdp, :rel_vol, :corr_yc, :corr_ytb, :corr_ysp, :def_prob]

    keys = [:mean_spr, :std_spr, :debt_gdp, :def_prob]

    Random.seed!(25)
    pv_uncond = simulvec(dd, uncond_K, burn_in=uncond_burn, Tmax=uncond_T, stopdef=false);

    pv = simulvec(dd, cond_K);

    moments = compute_moments(pv)
    moments[:def_prob] = mean(sum(pp[:def]) / max(sum(pp[:ζ].==1) / 4,0.25) * 100 for pp in pv_uncond)

    targets_vec = [targets[key] for key in keys]
    moments_vec = [moments[key] for key in keys]

    W = diagm(ones(4))
    # W[2,2] = 1 # More weight on std dev of spread

    showtable && table_moments(pv, pv_uncond, savetable = savetable)
    !showtable && smalltable && table_during(pv, pv_uncond, min_q)

    # names = [:sp, :std, :debt, :def]
    # indices= [1, 2, 3, 8]
    # dict = Dict(name => targets_vec[indices[jj]]-moments_vec[indices[jj]] for (jj, name) in enumerate(names))

    objective = (moments_vec ./ targets_vec .- 1)' * W * (moments_vec ./ targets_vec .- 1)
    objective, targets_vec, moments_vec#, dict
end

function simul_table(dd::DebtMod, dd_RE::DebtMod, K = 1_000; kwargs...)

    pv_uncond = simulvec(dd, 2_000, burn_in = 2_000, Tmax = 4_000, stopdef=false)
    pv = simulvec(dd, K)

    pv_uncond_RE = simulvec(dd_RE, 2_000, burn_in = 2_000, Tmax = 4_000, stopdef=false)
    pv_RE = simulvec(dd_RE, K)

    table_moments(pv, pv_uncond, pv_RE, pv_uncond_RE; kwargs...)
end

function solve_eval_α(dd::DebtMod, α)
    dd.pars[:α] = α
    Ny = length(dd.gr[:y])
    mpe!(dd)
    return dd.v[:V][1, ceil(Int, Ny / 2)]
end


function update_dd!(dd::DebtMod, params::Dict)
    for (key, val) in params
        if haskey(dd.pars, key)
            dd.pars[key] = val
        end
    end
end

function calibrate(dd::DebtMod, targets=PP_targets(); factor=0.1,
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

        w, t, m = calib_targets(dd, smalltable=false, cond_K=7_500, uncond_K=10_000)
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

# Trying with (β, d1, d2, θ) = (0.942, -0.216, 0.267, 0.609): ✓ (464) v = 18.9
# Trying with (β, d1, d2, θ) = (0.946, -0.206, 0.255, 0.669): ✓ (461) v = 17.4

#= This one goes instantly to one of the edges ??
function calibrate_SA(dd::DebtMod, targets = PP_targets();
    minβ = 1/(1+0.1),
    mind1 = -0.5,
    mind2 = 0.2,
    minθ = 0.005,
    maxβ = 1/(1+0.015),
    maxd1 = -0.01,
    maxd2 = 0.4,
    maxθ = 2,
    )

    keys = [:mean_spr, :std_spr, :debt_gdp, :rel_vol, :corr_yc, :corr_ytb, :corr_ysp, :def_prob]

    function objective(x)
        β = minβ + (maxβ - minβ)/(1+exp(-x[1]))
        d1 = mind1 + (maxd1 - mind1)/(1+exp(-x[2]))
        d2 = mind2 + (maxd2 - mind2)/(1+exp(-x[3]))
        θ = minθ + (maxθ - minθ)/(1+exp(-x[4]))

        dd.pars[:β] = β
        dd.pars[:d1] = d1
        dd.pars[:d2] = d2
        dd.pars[:θ] = θ

        print("Trying with (β, d1, d2, θ) = ($(@sprintf("%0.3g", β)), $(@sprintf("%0.3g", d1)), $(@sprintf("%0.3g", d2)), $(@sprintf("%0.3g", θ))): ")

        mpe!(dd, min_iter = 25, tol = 1e-6, tinyreport = true)

        w, t, m = calib_targets(dd, smalltable=false, cond_K = 7_500, uncond_K = 10_000)
        print("v = $(@sprintf("%0.3g", 100*w))\n")
        return w
    end

    xmin = [minβ, mind1, mind2, minθ]
    xmax = [maxβ, maxd1, maxd2, maxθ]
    xguess = [dd.pars[key] for key in [:β, :d1, :d2, :θ]]

    gβ = -log((maxβ - minβ)/(dd.pars[:β] - minβ) - 1)
    g1 = -log((maxd1 - mind1)/(dd.pars[:d1] - mind1) - 1)
    g2 = -log((maxd2 - mind2)/(dd.pars[:d2] - mind2) - 1)
    gθ = -log((maxθ - minθ)/(dd.pars[:θ] - minθ) - 1)

    xguess = [gβ, g1, g2, gθ]

    res = Optim.optimize(objective, xguess, SimulatedAnnealing())
end
=#
#=
function calib_sphere_ρ(dd::DebtMod;
    ρv, sρ=0.01, kwargs...)

    println("Now with ρ = $ρv")
    βv = (1+ρv)^-1
    βm = (1+ρv+sρ)^-1
    sβ = βv - βm

    calib_sphere(dd; βv = βv, sβ = sβ, kwargs...)
end

function calib_sphere(dd::DebtMod; W = Inf,
    βv = dd.pars[:β],
    d1v = dd.pars[:d1],
    d2v = dd.pars[:d2],
    θv = dd.pars[:θ],
    sβ = 0.001,
    sd1 = 0.01,
    sd2 = 0.01,
    sθ = 0.01
    )

    minβ = βv - sβ
    maxβ = βv + sβ
    mind1 = d1v - sd1
    maxd1 = d1v + sd1
    mind2 = d2v - sd2
    maxd2 = d2v + sd2
    minθ = θv - sθ
    maxθ = θv + sθ
    
    discrete_calibrate(dd, minβ=minβ, maxβ=maxβ, mind1=mind1, maxd1=maxd1, mind2=mind2, maxd2=maxd2, minθ=minθ, maxθ=maxθ, W=W)
end

function discrete_calibrate(dd::DebtMod;
    minβ = 1/(1+0.05),
    maxβ = 1/(1+0.03),
    mind1 = -0.23,
    maxd1 = -0.24,
    mind2 = 0.28,
    maxd2 = 0.29,
    minθ = 1.7,
    maxθ = 2,
    W = Inf,
    params = Dict(key => dd.pars[key] for key in [:β, :d1, :d2, :θ])
)

    gr_β = range(minβ, maxβ, length=11)
    gr_d1= range(mind1, maxd1, length=11)
    gr_d2= range(mind2, maxd2, length=11)
    gr_θ = range(minθ, maxθ, length=11)


    for (jβ, βv) in enumerate(gr_β), (jd1, d1v) in enumerate(gr_d1), (jd2, d2v) in enumerate(gr_d2), (jθ, θv) in enumerate(gr_θ)

        dd.pars[:β] = βv
        dd.pars[:d1]= d1v
        dd.pars[:d2]= d2v
        dd.pars[:θ] = θv

        print("Trying with (β, d1, d2, θ) = ($(@sprintf("%0.3g", βv)), $(@sprintf("%0.3g", d1v)), $(@sprintf("%0.3g", d2v)), $(@sprintf("%0.3g", θv)))\n")

        flag = mpe!(dd, tol = 1e-5, min_iter = 10, verbose = false)

        w, t, m, d = calib_targets(dd)

        print("\nCurrent W = $(@sprintf("%0.3g", w)), current best = $(@sprintf("%0.3g", W))\n")
        if w < W
            W = w

            for key in [:β, :d1, :d2, :θ]
                params[key] = dd.pars[key]
            end
        end
    end

    return params, W
end

function discrete_calib!(best_p, dd::DebtMod, maxiter = 200)
    iter = 0
    flag = false
    W = Inf

    while iter < maxiter
        iter += 1

        ρ = dd.pars[:β]^-1 - 1

        new_p, W = calib_sphere_ρ(dd, ρv=ρ, W=W)

        print("Best p so far: ")
        print(new_p)
        print("\n")

        if new_p == best_p
            break
        end

        for (key, val) in new_p
            best_p[key] = val
        end

        update_dd!(dd, new_p)
    end
end

function move_pars_solve!(dd, o_pars, sym, val)
    update_dd!(dd, o_pars)
    dd.pars[sym] = val
    println("Trying with (β, d1, d2, θ) = ($(@sprintf("%0.3g",dd.pars[:β])), $(@sprintf("%0.3g",dd.pars[:d1])), $(@sprintf("%0.3g",dd.pars[:d2])), $(@sprintf("%0.3g",dd.pars[:θ])))")
    mpe!(dd, min_iter = 10, tol = 1e-5, verbose = false)
    w, t, m, d = calib_targets(dd)
    return w,t,m,d
end

function gradient_ρ(dd::DebtMod;
    sρ=0.001,
    s1=0.001,
    s2=0.001,
    sθ=0.001
)
    pars = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ))

    mpe!(dd, min_iter = 10, tol = 1e-5, verbose = false)
    w, t, m, d = calib_targets(dd)

    ρv = dd.pars[:β]^-1 - 1
    ρ_up = ρv + sρ
    β_up = (1 + ρ_up)^-1

    d1_up = dd.pars[:d1] + s1
    d2_up = dd.pars[:d2] + s2
    θ_up = dd.pars[:θ] + sθ

    w_β,tβ,mβ,dβ = move_pars_solve!(dd, pars, :β, β_up)
    w_1,t1,m1,d1 = move_pars_solve!(dd, pars, :d1, d1_up)
    w_2,t2,m2,d2 = move_pars_solve!(dd, pars, :d2, d2_up)
    w_θ,tθ,mθ,dθ = move_pars_solve!(dd, pars, :θ, θ_up)

    d_β = w - w_β
    d_1 = w - w_1
    d_2 = w - w_2
    d_θ = w - w_θ

    vec = [d_β, d_1, d_2, d_θ]

    vec = vec / norm(vec) * norm([sρ, s1, s2, sθ])

    ρv += vec[1]
    pars[:β] = (1+ρv)^-1
    pars[:d1] += vec[2]
    pars[:d2] += vec[3]
    pars[:θ] += vec[4]

    return pars, w
end

function gradient_2(dd::DebtMod;
    s1=0.0025,
    s2=0.0025,
)
    pars = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ))

    mpe!(dd, min_iter = 10, tol = 1e-5, verbose = false)
    w, t0, m0, d = calib_targets(dd)

    w0 = (x->sum(x.^2))([t0[1], t0[3]*10] .- [m0[1], m0[3]*10])
    
    d1 = pars[:d1]
    d2 = pars[:d2]

    λ0 = d2 + d1
    λ1 = -d1

    λ0s = λ0 + s1
    λ1s = λ1 + s2

    d1_up = -λ1s
    d2_up = λ0s + λ1s

    w_1,t,m,d = move_pars_solve!(dd, pars, :d1, d1_up)
    w1 = (x->sum(x.^2))([t[1], t[3]*10] .- [m[1], m[3]*10])
    
    w_2,t,m,d = move_pars_solve!(dd, pars, :d2, d2_up)
    w2 = (x->sum(x.^2))([t[1], t[3]*10] .- [m[1], m[3]*10])

    vec = -[w0 - w1, w0 - w2]

    vec = vec / norm(vec) * norm([s1, s2])

    λ0 += vec[1]
    λ1 += vec[2]

    pars[:d1] = -λ1
    pars[:d2] = λ0 + λ1

    return pars, w0
end

function discrete_gradient!(best_p, dd::DebtMod, maxiter = 200)
    iter = 0
    W = Inf

    new_p = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ))
    currbest = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ))

    while iter < maxiter
        iter += 1

        for (key, val) in new_p
            currbest[key] = val
        end

        new_p, w = gradient_2(dd)
        
        print("\nCurrent inner W = $(@sprintf("%0.3g", w)), current best = $(@sprintf("%0.3g", W))\n")

        if w < W
            for (key, val) in currbest
                best_p[key] = val
            end
        else
            break
        end
        print("Best p so far: ")
        print(currbest)
        print("\n")
      
        W = w
        update_dd!(dd, new_p)
    end
    update_dd!(dd, best_p)
end

function eval_nested!(dd, o_pars, sym, val, n_pars)
    update_dd!(dd, o_pars)
    dd.pars[sym] = val

    discrete_gradient!(n_pars, dd)

    println("Trying outer with (β, d1, d2, θ) = ($(@sprintf("%0.3g",dd.pars[:β])), $(@sprintf("%0.3g",dd.pars[:d1])), $(@sprintf("%0.3g",dd.pars[:d2])), $(@sprintf("%0.3g",dd.pars[:θ])))")
    mpe!(dd, min_iter=10, tol=1e-5, verbose=false)
    w, t, m, d = calib_targets(dd)
    return w,t,m,d
end


function gradient_nested(dd::DebtMod;
    sρ=0.005,
    sθ=0.005
)

    o_pars = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ))
    n_pars = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ))

    # mpe!(dd, min_iter=10, tol=1e-5, verbose=false)
    # w, t, m, d = calib_targets(dd)
    # w = (x -> sum(x.^2))([t[2], t[end]] - [m[2], m[end]])
    w, t, m, d = eval_nested!(dd, o_pars, :β, o_pars[:β], n_pars)
    w = (x -> sum(x.^2))([t[1], t[2], t[3]*10, t[end]*25] - [m[1], m[2], m[3]*10, m[end]*25])
    pars_center = Dict(key => val for (key, val) in n_pars)

    ρv = o_pars[:β]^-1 - 1

    ρ1 = ρv + sρ
    θ1 = o_pars[:θ] + sθ

    β1 = (1 + ρ1)^-1

    wβ, t, m, d = eval_nested!(dd, o_pars, :β, β1, n_pars)
    wβ = (x -> sum(x.^2))([t[1], t[2], t[3]*10, t[end]*25] - [m[1], m[2], m[3]*10, m[end]*25])

    wθ, t, m, d = eval_nested!(dd, o_pars, :θ, θ1, n_pars)
    wθ = (x -> sum(x.^2))([t[1], t[2], t[3]*10, t[end]*25] - [m[1], m[2], m[3]*10, m[end]*25])

    vec = [w - wβ, w - wθ]
    vec = vec / norm(vec) * norm([sρ, sθ])

    ρ1 += vec[1]
    o_pars[:β] = (1 + ρ1)^-1
    o_pars[:θ] += vec[2]

    return o_pars, w, pars_center
end


function discrete_gradient_nested!(best_p, dd::DebtMod, maxiter = 100)
    iter = 0
    W = Inf

    new_p = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ))

    while iter < maxiter
        iter += 1

        new_p, w, pars_opt = gradient_nested(dd)

        print("\nCurrent outer W = $(@sprintf("%0.3g", w)), current best = $(@sprintf("%0.3g", W))\n")

        if w < W
        else
            break
        end
        
        for (key, val) in pars_opt
            best_p[key] = val
        end

        print("Best p so far: ")
        print(best_p)
        print("\n")

        W = w
        update_dd!(dd, new_p)
    end

    print("\nOuter iters: $iter")
    update_dd!(dd, best_p)
end
=#

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

function eval_Sobol(dd::DebtMod, key, val, verbose)
    setval!(dd.pars, key, val)

    mpe!(dd, min_iter = 25, maxiter = 1_000, tol = 1e-6, tinyreport = true)
    mpe!(dd, min_iter = 25, maxiter = 1_000, tol = 1e-6, tinyreport = true)
    mpe!(dd, min_iter = 25, maxiter = 1_000, tol = 1e-6, tinyreport = true)
    w, t, m = calib_targets(dd, smalltable=verbose, cond_K = 7_500, uncond_K = 10_000)
    verbose || print("w=$(@sprintf("%0.3g", 100*w)) ")
    w
end

function iter_Sobol(dd::DebtMod, key, σ; Nx = 15)
    
    x = getval(key, dd.pars)
    xvec = range(x-σ, x+σ, length = Nx)
    jc = ceil(Int, Nx/2)

    W = Inf
    xopt = x
    jopt = jc
    
    for (jx, xv) in enumerate(xvec)

        w = eval_Sobol(dd, key, xv, (jx==jc))
        jx == jc && print("w = $(@sprintf("%0.3g", 100*w))\n")

        if w < W
            W = w
            xopt = xv
            jopt = jx
        end
    end

    return xopt, jopt, W, x
end


function pseudoSobol!(dd::DebtMod, best_p = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ));
    maxiter = 500,
    σβ = 0.0001, σθ = 0.005, σ1 = 0.0001, σ2 = 0.00002)
    
    update_dd!(dd, best_p)

    σvec = [σβ, σθ, σ1, σ2]
    names = [:β, :θ, :d1, :d2]
    Nxs = [5,5,5,5]
    
    curr_p = Dict(key => dd.pars[key] for key in (:β, :d1, :d2, :θ))
    W = Inf

    iter = 0
    while iter < maxiter
        iter += 1

        js = iter % 4
        js == 0 ? js = 4 : nothing

        key = names[js]
        σ = σvec[js]/2

        Nx = Nxs[js]

        print("Iteration $iter at $(Dates.format(now(), "HH:MM")). Moving $key from $(curr_p)\n")
        xopt, jopt, w, x_og = iter_Sobol(dd, key, σ, Nx=Nx)
        
        print("\nBest objective: $(@sprintf("%0.3g", 100*w)) at $key [$jopt] = $(@sprintf("%0.5g", xopt)) in [$(@sprintf("%0.5g", x_og-σ)), $(@sprintf("%0.5g", x_og+σ))]. ")
        
        setval!(curr_p, key, xopt)
        mpe!(dd, min_iter = 25, maxiter = 1_000, tol = 5e-7, tinyreport = true)

        if w < W
            W = w
            for (key, val) in curr_p
                best_p[key] = val
            end
        end
        print("Best so far $(@sprintf("%0.3g", 100*W))\n")
        
        update_dd!(dd, curr_p)
    end
end