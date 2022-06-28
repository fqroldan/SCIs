## TO DO: one long simul to get def.prob, then 1000 simuls of 200+whatever is needed to get 35+4 periods without default followed by a default. ✓

## TO DO: outside of simul, compute q* for an undefaulteable bond for given r in range(0.01, 0.21, length > 400), then use this to interpolate r for the bond in the model to get the spread. ✓

## TO DO: compute std for log(c), log(y), b = B/Y (not 4Y), correlations standard with variables in levels not logs. ✓

## TO DO: sit down with FRoch for the DEP calculations

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

Base.getindex(pp::SimulPath, s::Symbol) = pp.data[:, pp.names[s]]

subpath(pp::SimulPath, t0, t1) = SimulPath(pp.names, pp.data[t0:t1, :])


function iter_simul(b0, y0, def::Bool, itp_R, itp_D, itp_prob, itp_c, itp_b, itp_q, itp_qD, itp_yield, pars::Dict, min_y, max_y)
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
    
    ϵ = rand(Normal(0, 1))
    yp = exp(ρy * log(y0) + σy * ϵ)

    yp = min(max_y, max(min_y, yp))

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

    return v, def_p, ct, bp, yp, new_def, q, spread
end

function simulvec(dd::DebtMod, K; burn_in=200, Tmax=10 * burn_in, cond_defs = 35, separation = 4, stopdef = true)

    cd_sep = cond_defs + separation

    pv = Vector{SimulPath}(undef, K)

    itp_yield = get_yields_itp(dd)

    for jp in eachindex(pv)
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
    
        b0 = 0.0
        y0 = mean(dd.gr[:y])
        def = false
        new_def = false
    
        min_y, max_y = extrema(dd.gr[:y])
    
        pp = SimulPath(Tmax, [:c, :b, :y, :v, :ζ, :v_cond, :def, :q, :spread, :sp])
    
        contflag = true
    
        t = 0
        while contflag && t < Tmax
            t += 1
    
            pp[:y, t] = y0
            pp[:b, t] = b0
    
            pp[:ζ, t] = ifelse(def, 2, 1)
            pp[:def, t] = ifelse(new_def, 1, 0)
    
            pp[:v, t] = itp_v(b0, y0)
    
            v, def, ct, b0, y0, new_def, q, spread = iter_simul(b0, y0, def, itp_R, itp_D, itp_prob, itp_c, itp_b, itp_q, itp_qD, itp_yield, dd.pars, min_y, max_y)
    
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

    moments[:debt_gdp] = mean(mean(pp[:b]./pp[:y]) * 100 for pp in pv)

    moments[:rel_vol]  = mean(std(log.(pp[:c])) ./ std(log.(pp[:y])) for pp in pv)

    moments[:corr_yc]  = mean(cor(pp[:c], pp[:y]) for pp in pv)

    moments[:corr_ytb] = mean(cor(pp[:y], pp[:y] .- pp[:c]) for pp in pv)

    moments[:corr_ysp] = mean(cor(pp[:y], pp[:spread]) for pp in pv)

    return moments
end

function PP_targets()
    targets = Dict{Symbol,Float64}(
        :mean_spr => 815,
        :mean_sp  => 815,
        :std_spr => 458,
        :debt_gdp => 46,
        :rel_vol => 0.87,
        :corr_yc => 0.97,
        :corr_ytb => -0.77,
        :corr_ysp => -0.72,
        :def_prob => 3
    )
end

function table_moments(pv::Vector{SimulPath}, pv_uncond::Vector{SimulPath}, pv_RE=[], pv_uncond_RE=[]; savetable=false)

    # println(length(pv_RE))

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

function calib_targets(dd::DebtMod; cond_K = 1_000, uncond_K = 2_000 , uncond_burn = 2_000, uncond_T = 4_000)
    targets = PP_targets()

    keys = [:mean_spr,
    # :mean_sp,
    :std_spr, :debt_gdp, :rel_vol, :corr_yc, :corr_ytb, :corr_ysp, :def_prob]

    Random.seed!(25)
    pv_uncond = simulvec(dd, uncond_K, burn_in=uncond_burn, Tmax=uncond_T, stopdef=false)
    pv = simulvec(dd, cond_K)

    moments = compute_moments(pv)
    moments[:def_prob] = mean(sum(pp[:def]) / (sum(pp[:ζ].==1) / 4) * 100 for pp in pv_uncond)

    targets_vec = [targets[key] for key in keys]
    moments_vec = [moments[key] for key in keys]

    W = ones(length(keys), length(keys))

    table_moments(pv, pv_uncond, savetable = false)

    objective = (targets_vec - moments_vec)' * W * (targets_vec - moments_vec)
    objective, targets_vec, moments_vec
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


function calibrate(dd::DebtMod, targets = PP_targets();
    minβ = 1/(1+0.1),
    mind1 = -0.5,
    mind2 = 0.2,
    minθ = 0.5,
    maxβ = 1/(1+0.015),
    maxd1 = -0.01,
    maxd2 = 0.4,
    maxθ = 3,
)
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

        println("Trying with (β, d1, d2, θ) = ($β, $d1, $d2, $θ)")

        mpe!(dd, min_iter = 10, verbose=false)

        w, t, m = calib_targets(dd)
        
        return w
    end

    xmin = [minβ, mind1, mind2, minθ]
    xmax = [maxβ, maxd1, maxd2, maxθ]
    xguess = [dd.pars[key] for key in [:β, :d1, :d2, :θ]]

    res = Optim.optimize(objective, xmin, xmax, xguess, Fminbox(NelderMead()))
end
    
function discrete_calibrate(dd::DebtMod, targets = PP_targets();
    minβ = 1/(1+0.045),
    maxβ = 1/(1+0.03),
    mind1 = -0.275,
    maxd1 = -0.235,
    mind2 = 0.25,
    maxd2 = 0.35,
    minθ = 0.5,
    maxθ = 3,
)

    gr_β = range(minβ, maxβ, length=10)
    gr_d1= range(mind1, maxd1, length=10)
    gr_d2= range(mind2, maxd2, length=10)
    gr_θ = range(minθ, maxθ, length=10)

    W = Inf
    params = Dict(key => dd.pars[key] for key in [:β, :d1, :d2, :θ])

    for (jβ, βv) in enumerate(gr_β), (jd1, d1v) in enumerate(gr_d1), (jd2, d2v) in enumerate(gr_d2), (jθ, θv) in enumerate(gr_θ)

        dd.pars[:β] = βv
        dd.pars[:d1]= d1v
        dd.pars[:d2]= d2v
        dd.pars[:θ] = θv

        print("Trying with (β, d1, d2, θ) = ($(@sprintf("%0.3g", βv)), $(@sprintf("%0.3g", d1v)), $(@sprintf("%0.3g", d2v)), $(@sprintf("%0.3g", θv)))\n")

        flag = mpe!(dd, min_iter = 10, verbose = false)

        w, t, m = calib_targets(dd)

        if flag && w < W
            W = w

            for key in [:β, :d1, :d2, :θ]
                params[key] = dd.pars[key]
            end
        end
    end

    return params
end