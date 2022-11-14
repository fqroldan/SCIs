function comp_argbond(dd::DebtMod; showtable=false, smalltable=!showtable, DEP=false)
    @assert dd.pars[:α] == 0 && dd.pars[:τ] <= minimum(dd.gr[:y])

    mpe_simul!(dd, maxiter=500, K=0, simul=false)

    Random.seed!(25)
    w, t, m, ϵvv, ξvv = calib_targets(dd, smalltable=smalltable, showtable=showtable, uncond_K=10_000)


    Ny = length(dd.gr[:y])
    v_noncont = welfare(dd)
    c_noncont = cons_equiv(v_noncont, dd)
    print("V with noncont: $v_noncont, c = $c_noncont\n")

    if DEP
        DEP_bylength(dd, [35/4, 60])
    end

    dd.pars[:α] = 1
    dd.pars[:min_q] = 0

    mpe_simul!(dd, maxiter=500, K=8, simul=false)
    calib_targets(dd, ϵvv, ξvv, uncond_K=10_000, smalltable=smalltable, showtable=showtable)

    v_linear = welfare(dd)
    c_linear = cons_equiv(v_linear, dd)
    print("V with linear: $v_linear, c = $c_linear. ")
    gains_linear = c_linear / c_noncont - 1
    print("Gains with linear = $(@sprintf("%0.3g", 100*gains_linear))%\n")

    if DEP
        DEP_bylength(dd, [35/4, 60])
    end

    dd.pars[:τ] = 1
    dd.gr[:b] = collect(range(0, 2 * maximum(dd.gr[:b]), length=length(dd.gr[:b])))

    mpe_simul!(dd, maxiter=500, K=8, simul=false)
    calib_targets(dd, ϵvv, ξvv, uncond_K=10_000, smalltable=smalltable, showtable=showtable)

    v_threshold = welfare(dd)
    c_threshold = cons_equiv(v_threshold, dd)
    print("V with threshold: $v_threshold, c = $c_threshold. ")
    gains_threshold = c_threshold / c_noncont - 1
    print("Gains with threshold = $(@sprintf("%0.3g", 100*gains_threshold))%\n")
    if DEP
        DEP_bylength(dd, [35/4, 60])
    end

    return v_noncont, v_linear, v_threshold
end

function comp_t5(dd::DebtMod; showtable=false)
    @assert dd.pars[:θ] >= 1e-3

    print("Solving original model with θ = $(dd.pars[:θ])\n")
    save("dd_comp_theta.jld2", "dd", dd)
    rob_n, rob_l, rob_t = comp_argbond(dd, DEP = true, showtable=showtable)

    dd = load("dd_comp_theta.jld2", "dd")
    dd.pars[:θ] = 0
    print("Solving same model with rational expectations\n")
    mpe_simul!(dd, maxiter=500, K=8, simul=false)
    rat_n, rat_l, rat_t = comp_argbond(dd, showtable=showtable)

    return rob_n, rob_l, rob_t, rat_n, rat_l, rat_t
end

function DEP_bylength(dd::DebtMod, Yvec = range(10, 60, length=6))

    for years in Yvec
        _, DEP = simul_dist(dd, K = 2_000, burn_in = 2_000, T = Int(4years))
        print("DEP computed on $years years ($(Int(4years)) periods): $(@sprintf("%0.3g", 100*DEP))%\n")
    end
end

function try_all_SCIs(dd::DebtMod; Nτ = 15)
    @assert dd.pars[:α] == 0 && dd.pars[:τ] <= minimum(dd.gr[:y])

    αvec = 0:0.5:10
    Nα = length(αvec)
    τvec = range(minimum(dd.gr[:y]), mean(dd.gr[:y]), length=Nτ)

    Vs = zeros(Nα, Nτ)
    Cs = zeros(Nα, Nτ)

    N = length(Vs)
    n = 0
    print("Starting evaluation of all SCIs. ")
    c_baseline = 0.
    c_star, α_star, τ_star = -Inf, 0., 0.
    c_gain_star = 0.

    dd.gr[:b] = collect(range(0, 2 * maximum(dd.gr[:b]), length=length(dd.gr[:b])))

    for (jα, αv) in enumerate(αvec), (jτ, τv) in enumerate(τvec)
        n += 1
        dd.pars[:α] = αv
        dd.pars[:τ] = τv
        print("Solving with α = $αv, τ = $(@sprintf("%0.3g", τv)) ($n/$N) at $(Dates.format(now(), "HH:MM"))\n")

        mpe_simul!(dd, maxiter=500, K=4, tol = 2.5e-6, simul=false)
        v = welfare(dd)
        c = cons_equiv(v, dd)
        if n == 1
            c_baseline = c
        end
        c_gain = 100*(c/c_baseline-1)
        if c > c_star
            c_star = c
            α_star = αv
            τ_star = τv
            c_gain_star = c_gain
        end

        print("Cons equiv: $(@sprintf("%0.3g", c)), $(@sprintf("%0.3g", c_gain))% from baseline \n")
        print("Best so far $(@sprintf("%0.3g", c_gain_star))% with (α, τ) = ($(@sprintf("%0.3g", α_star)), $(@sprintf("%0.3g", τ_star)))\n")
        Vs[jα, jτ] = v
        Cs[jα, jτ] = c
    end

    return Vs, Cs, αvec, τvec
end

function save_all_SCIs(dd::DebtMod)
    V, C, αvec, τvec = try_all_SCIs(dd)

    dd.pars[:θ] == 0 ? rob = "RE" : rob = "rob"
    h = dd.pars[:ℏ]

    save("comp_alphatau_h$(h)_$rob.jld2", "V", V, "C", C, "αvec", αvec, "τvec", τvec)
end

welfare(dd::DebtMod) = dot(dd.v[:V][1, :], stationary_distribution(dd))

function compare_bonds(dd::DebtMod, α1, τ1, αRE, τRE)

    fy = stationary_distribution(dd)

    R_RE = (dd.gr[:y].>= τRE) .* (max.(0, 1 .+ αRE * (dd.gr[:y] .- 1)))
    R_Rob = (dd.gr[:y].>= τ1) .* (max.(0, 1 .+ α1 * (dd.gr[:y] .- 1)))

    println(typeof(R_RE))
    max_R = max(maximum(R_RE), maximum(R_Rob))

    lines = [
        scatter(x=dd.gr[:y], y=R_RE, name = "RE")
        scatter(x=dd.gr[:y], y=R_Rob, name = "Robustness")
        bar(x=dd.gr[:y], y = fy ./ maximum(fy) .* max_R, opacity=0.25, name = "Ergodic dist")
    ]

    plot(
        lines,
        Layout(
            title="Optimal bond structure", xaxis_title="<i>y",
            font="Lato", fontsize=18,
            xaxis = attr(showgrid = true, gridcolor="#e9e9e9", gridwidth=0.5, zeroline=false),
            yaxis = attr(showgrid = true, gridcolor="#e9e9e9", gridwidth=0.5, zeroline=false),
            legend=attr(orientation="h", x=0.05, xanchor="left")
        )
    )
end