abstract type DebtMod end

struct Default <: DebtMod
    pars::Dict{Symbol,Float64}
    gr::Dict{Symbol,Vector{Float64}}

    P::Dict{Symbol,Matrix{Float64}}

    v::Dict{Symbol,Matrix{Float64}}
    
    vL::Array{Float64, 3}

    gc::Array{Float64,3}
    gb::Array{Float64,2}

    q::Matrix{Float64}
    qD::Matrix{Float64}
end

# κ = 0.0785 = λ + (1-λ) z = 0.05 + 0.95*0.03 in CE
CE_params(;kwargs...) = Default(γ = 2, σy = 0.027092, ρy = 0.948503, ψ = 0.0385, r = 0.01, ρ = 0.05, β = 0.95402, d1 = -0.18819, d2 = 0.24558, κ = 0.0785; kwargs...)

function Default(;
    β=0.9627,
    γ=2,
    r=0.01,
    θ=1.6155,
    ψ=0.0385,
    χ = 0.01,
    α=0,
    τ=0,
    ρ=0.05,
    # ℏ=1.0,
    ℏ = 0.4,
    d1 = -0.255,
    d2 = 0.296,
    ρy=0.9484,
    σy=0.02,
    Nb=150,
    Ny=91,
    bmax=1.25,
    std_devs = 5,
    κ = r+ρ,
    min_q = 0.35,
)

    βL = 1/(1+r)
    wL = 1

    pars = Dict(:β => β, :γ => γ, :r => r, :θ => θ, :χ => χ, :ρ => ρ, :κ => κ, :ℏ => ℏ, :d1 => d1, :d2 => d2, :ρy => ρy, :σy => σy, :α => α, :τ => τ, :ψ => ψ, :βL => βL, :wL => wL, :min_q => min_q, :Nb => Nb, :Ny => Ny, :bmax => bmax, :std_devs => std_devs)

    ychain = tauchen(Ny, ρy, σy, 0, std_devs)

    Py = ychain.p
    ygrid = exp.(ychain.state_values)

    bgrid = range(0, bmax, length=Nb)

    gr = Dict(:b => bgrid, :y => ygrid)
    P = Dict(:y => Py)

    v = Dict(key => zeros(Nb, Ny) for key in (:V, :R, :D, :prob))

    gc = zeros(Nb, Ny, 2)
    gb = zeros(Nb, Ny)

    vL = zeros(Nb, Ny, 2)

    q = ones(Nb, Ny)
    qD = zeros(Nb, Ny)

    return Default(pars, gr, P, v, vL, gc, gb, q, qD)
end

info(dd::DebtMod) = Dict(
    key => dd.pars[key] for key in (:β, :θ, :d1, :d2, :α, :τ, :ℏ, :χ, :κ, :Nb, :Ny, :bmax, :std_devs) if haskey(dd.pars, key)
)

cons_equiv(v::Number, dd::DebtMod) = cons_equiv(v, dd.pars[:β], dd.pars[:γ])
cons_equiv(v::Number, β::Number, γ::Number) = (v * (1-β) * (1-γ))^(1/(1-γ))

function logsumexp(a::AbstractVector{<:Real})
    m = maximum(a)
    return m + log.(sum(exp.(a .- m)))
end

N(dd::DebtMod, sym::Symbol) = length(dd.gr[sym])

function make_itp(dd::DebtMod, y::Array{<:Number, 2})
    knots = (dd.gr[:b], 1:N(dd, :y))

    interpolate(knots, y, (Gridded(Linear()), NoInterp()))
end

function make_itp(dd::DebtMod, y::Array{<:Number, 3})
    knots = (dd.gr[:b], 1:N(dd, :y), 1:2)

    interpolate(knots, y, (Gridded(Linear()), NoInterp(), NoInterp()))
end


u(cv, dd::DebtMod) = u(cv, dd.pars[:γ])
function u(cv, γ)
    cmin = 1e-3
    if cv < cmin
        # Below cmin, linear continuation using derivative
        return u(cmin, γ) + (cv - cmin) * cmin^(-γ)
    else
        if γ == 1
            return log(cv)
        else
            return cv^(1 - γ) / (1 - γ)
        end
    end
end

cons_in_default(yv, dd::DebtMod) = cons_in_default(yv, dd.pars[:d1], dd.pars[:d2])
cons_in_default(yv, d1::Number, d2::Number) = yv - max(0, d1 * yv + d2 * yv^2)

function coupon_rate(yv, dd::DebtMod; α = dd.pars[:α], τ = dd.pars[:τ])
    κ = dd.pars[:κ]

    linear_coup = 1 + α * (yv-1)
    if yv >= τ
        return max(0.0, κ * linear_coup)
    else
        return 0.0
    end
end

function budget_constraint(bpv, bv, yv, q, dd::DebtMod)
    ρ = dd.pars[:ρ]

    coupon = coupon_rate(yv, dd)
    
    # consumption equals income plus proceeds from selling debt minus old debt repayments
    cv = yv + q * (bpv - (1 - ρ) * bv) - coupon * bv
    return cv
end

function BC(bpv, jb, jy, itp_q, dd::DebtMod)
    bv, yv = dd.gr[:b][jb], dd.gr[:y][jy]

    # Interpolate debt price at chosen issuance level
    qv = itp_q(bpv, jy)

    cv = budget_constraint(bpv, bv, yv, qv, dd)
end

function eval_value(jb, jy, bpv, itp_q, itp_Ev, dd::DebtMod)
    """ Evalúa la función de valor en (b,y) para una elección de b' """
    β = dd.pars[:β]
    
    cv = BC(bpv, jb, jy, itp_q, dd)
    ut = u(cv, dd)

    Ev = itp_Ev(bpv)

    v = ut + β * Ev

    return v, cv
end

function max_adj(jb, jy, bmax, itp_q, dd::DebtMod)
    bmin = minimum(dd.gr[:b])

    objf(bpv) = BC(bpv, jb, jy, itp_q, dd)

    if objf(bmin) < 0
        res = Optim.optimize(x -> objf(x)^2, bmin, bmax, GoldenSection())
        bmin = res.minimizer
    end
    return bmin
end

function opt_value(jb, jy, bmax, itp_q, itp_Ev, dd::DebtMod)
    """ Elige b' en (b,y) para maximizar la función de valor """

    # b' ∈ bgrid
    bmin = max_adj(jb, jy, bmax, itp_q, dd)

    # Objective function in terms of b', with minus to minimize
    obj_f(bpv) = -eval_value(jb, jy, bpv, itp_q, itp_Ev, dd)[1]

    if bmax - bmin < 1e-4
        b_star = bmax
    else
        # Get max
        res = Optim.optimize(obj_f, bmin, bmax, GoldenSection())
        # Get argmax
        b_star = res.minimizer
    end

    # Get v, c consistent with b'
    vp, c_star = eval_value(jb, jy, b_star, itp_q, itp_Ev, dd)

    return vp, c_star, b_star
end

function value_default(jb, jy, dd::DebtMod)
    β, ψ = (dd.pars[sym] for sym in (:β, :ψ))
    """ Value of default at (b,y) """
    yv = dd.gr[:y][jy]

    c = cons_in_default(yv, dd)

    c > 1e-2 || println("WARNING: negative c at (jb, jy) = ($jb, $jy)")

    # Continuation value includes chance of reaccessing markets
    Ev = 0.0
    for jyp in eachindex(dd.gr[:y])
        prob = dd.P[:y][jy, jyp]
        Ev += prob * (ψ * dd.v[:V][jb, jyp] + (1 - ψ) * dd.v[:D][jb, jyp])
    end

    v = u(c, dd) + β * Ev

    return c, v
end

function borrowing_limit_q(jy, itp_q, dd::DebtMod)
    bmin, bmax = extrema(dd.gr[:b])

    min_q = dd.pars[:min_q]

    objf(bpv) = (itp_q(bpv, jy) - min_q)^2

    if itp_q(bmax, jy) < min_q
        res = Optim.optimize(objf, bmin, bmax, GoldenSection())
        bmax = res.minimizer
    end
    return bmax
end

function borrowing_limit(itp_def, dd::DebtMod)
    bmin, bmax = extrema(dd.gr[:b])

    max_prob = 0.95

    objf(bpv) = (itp_def(bpv) - max_prob)^2

    if itp_def(bmax) > max_prob
        res = Optim.optimize(objf, bmin, bmax, GoldenSection())
        bmax = res.minimizer
    end
    return bmax
end

function make_vp(dd, jy, mat)
    Ev = similar(dd.gr[:b])
    for jbp in eachindex(dd.gr[:b])
        Evc = 0.0
        for jyp in eachindex(dd.gr[:y])
            prob = dd.P[:y][jy,jyp]

            Evc += prob * mat[jbp, jyp]
        end
        Ev[jbp] = Evc
    end
    return Ev
end

function make_itp_vp(dd::DebtMod, jy, mat)
    Ev = make_vp(dd, jy, mat)
    interpolate((dd.gr[:b],), Ev, Gridded(Linear()))
end

function vfi_iter!(new_v, itp_q, dd::DebtMod)
    
    Threads.@threads for jy in eachindex(dd.gr[:y])
        itp_Ev = make_itp_vp(dd, jy, dd.v[:V]);
        itp_def = make_itp_vp(dd, jy, dd.v[:prob]);

        bmax = borrowing_limit(itp_def, dd)
        for jb in eachindex(dd.gr[:b])
        
            # Repayment
            vp, c_star, b_star = opt_value(jb, jy, bmax, itp_q, itp_Ev, dd)

            # Save values for repayment 
            dd.v[:R][jb, jy] = vp
            dd.gb[jb, jy] = b_star
            dd.gc[jb, jy, 1] = c_star

            # In default
            cD, vD = value_default(jb, jy, dd)
            dd.v[:D][jb, jy] = vD
            dd.gc[jb, jy, 2] = cD
        end
    end

    χ, ℏ = (dd.pars[sym] for sym in (:χ, :ℏ))
    itp_vD = make_itp(dd, dd.v[:D]);
    Vs = zeros(2)
    for (jb, bv) in enumerate(dd.gr[:b]), jy in eachindex(dd.gr[:y])
        # Value of repayment and default at (b,y)
        vr = dd.v[:R][jb, jy]
        vd = itp_vD((1 - ℏ) * bv, jy)

        Vs .= (vd, vr)

        ## Numerically-stable log-sum-exp for value function (above eq. 4)
        lse = logsumexp(Vs ./ χ)
        lpr = vd / χ - lse
        pr = exp(lpr)
        V = χ * lse

        # Save value function and ex-post default probability
        new_v[jb, jy] = V
        dd.v[:prob][jb, jy] = pr
    end
end

function logsumexp_onepass(X, w)
    a = -Inf
    r = zero(eltype(X))
    for (x, wi) in zip(X, w)
        if x <= a
            # standard computation
            r += wi * exp(x - a)
        else
            # if new value is higher than current max
            r *= exp(a - x)
            r += wi
            a = x
        end
    end
    return a + log(r)
end

function value_lenders(bv, bpv, jy, py, coupon, itp_q, itp_def, itp_vL, dd::DebtMod, vLp; rep)
    θ, wL, βL, ρ, ψ, ℏ = (dd.pars[sym] for sym in (:θ, :wL, :βL, :ρ, :ψ, :ℏ))
    
    cL = wL
    if rep
        cL += coupon * bv - itp_q(bpv, jy) * (bpv - (1 - ρ) * bv)
    end

    if !rep
        bpv = bv
    end

    # With robustness to preference shock (need to uncomment gr, w, x in v_lender_iter below and pass as arguments)
    # for js in axes(gr, 1)
    #     jyp, jζp = gr[js, :]   # jζp = 1 in rep, 2 in def

    #     p_def = ifelse(rep, itp_def(bpv, jyp), 1-ψ)
    #     prob = py[jyp]

    #     w[js] = ifelse(jζp == 1, 1-p_def, p_def) * prob

    #     # haircut when going from repayment to default
    #     b_pv = ifelse(rep && jζp == 2, (1-ℏ)*bpv, bpv) 

    #     x[js] = -θ * itp_vL(b_pv, jyp, jζp)
    # end
    # # log ∑_i prob_i exp(-θ v^L_i)
    # Tv = logsumexp_onepass(x, w) / -θ

    # Without robustness to the preference shock
    for jyp in eachindex(dd.gr[:y])

        p_def = ifelse(rep, itp_def(bpv, jyp), 1-ψ)
        bpv_R = bpv
        bpv_D = ifelse(rep, (1-ℏ)*bpv, bpv)

        vLp[jyp] = p_def * itp_vL(bpv_D, jyp, 2) + (1-p_def) * itp_vL(bpv_R, jyp, 1)
    end
    Tv = logsumexp_onepass(-θ * vLp, py) / -θ

    vL = cL + βL * Tv

    return vL
end

function v_lender_iter!(dd::Default, vL = dd.vL, q = dd.q, α = dd.pars[:α], τ = dd.pars[:τ])
    itp_q = make_itp(dd, q);
    itp_def = make_itp(dd, dd.v[:prob]);

    itp_vL = make_itp(dd, vL)

    if dd.pars[:θ] > 1e-3
        # gr = gridmake(1:length(dd.gr[:y]), 1:2)
        Threads.@threads for jy in eachindex(dd.gr[:y])
            yv = dd.gr[:y][jy]
            coupon = coupon_rate(yv, dd, α = α, τ = τ)

            py = dd.P[:y][jy, :]

            # w = zeros(size(gr,1))
            # x = zeros(size(gr,1))
            vLp = similar(dd.gr[:y])
            
            for (jb, bv) in enumerate(dd.gr[:b])

                bpv = dd.gb[jb, jy]
                vL[jb, jy, 1] = value_lenders(bv, bpv, jy, py, coupon, itp_q, itp_def, itp_vL, dd, vLp, rep=true)
                vL[jb, jy, 2] = value_lenders(bv, bpv, jy, py, coupon, itp_q, itp_def, itp_vL, dd, vLp, rep=false)
            end
        end
    else
        vL[:] .= 1
    end
    nothing
end

function q_iter!(new_q, new_qd, dd::Default)
    """ Lenders' Euler equation determines debt prices given debt, income, and expected debt prices """
    ρ, ℏ, ψ, r, θ = (dd.pars[sym] for sym in (:ρ, :ℏ, :ψ, :r, :θ))

    # Interpolate debt prices for next period
    itp_qd = make_itp(dd, dd.qD)
    itp_q = make_itp(dd, dd.q)

    for (jbp, bpv) in enumerate(dd.gr[:b]), (jy, yv) in enumerate(dd.gr[:y])
        Eq = 0.0
        EqD = 0.0
        sum_sdf_R = 0.0
        sum_sdf_D = 0.0
        for (jyp, ypv) in enumerate(dd.gr[:y])
            prob_def = dd.v[:prob][jbp, jyp]

            if θ > 1e-3
                sdf_R = exp(-θ * dd.vL[jbp, jyp, 1])
                sdf_D = exp(-θ * dd.vL[jbp, jyp, 2])
            else
                sdf_R = 1.0
                sdf_D = 1.0
            end

            coupon = coupon_rate(ypv, dd)

            # If access to markets, new issuance and can default tomorrow
            bpp = dd.gb[jbp, jyp]
            rep_R = (1 - prob_def) * sdf_R * (coupon + (1 - ρ) * itp_q(bpp, jyp)) + prob_def * sdf_D * (1 - ℏ) * itp_qd((1 - ℏ) * bpv, jyp)

            # If default, may reaccess markets
            rep_D = ψ * rep_R + (1 - ψ) * sdf_D * dd.qD[jbp, jyp]

            prob = dd.P[:y][jy, jyp]
            Eq += prob * rep_R
            EqD += prob * rep_D

            sum_sdf_R += prob * (prob_def * sdf_D + (1 - prob_def) * sdf_R)
            sum_sdf_D += prob * ((1 - ψ) * sdf_D + ψ * sdf_R)
        end
        new_q[jbp, jy] = Eq / (1 + r) / sum_sdf_R
        new_qd[jbp, jy] = EqD / (1 + r) / sum_sdf_D
    end
end

function mpe!(dd::Default; tol=1e-6, maxiter=500, upd_η = 1., min_iter = 1, tinyreport::Bool = false, verbose = !tinyreport)

    new_v = similar(dd.v[:V]);
    new_q = similar(dd.q);
    new_qd = similar(dd.qD);

    dist = 1 + tol
    iter = 0

    while iter < min_iter || (dist > tol && iter < maxiter)
        iter += 1

        iter % 100 == 0 && print(".")

        verbose && print("Iteration $iter: ")

        # Update debt prices
        v_lender_iter!(dd)
        q_iter!(new_q, new_qd, dd)
        dist_qR = norm(new_q - dd.q) / max(1, norm(dd.q))
        dist_qD = norm(new_qd - dd.qD)/max(1, norm(dd.qD))
        dist_q = max(dist_qD, dist_qR)

        # Interpolate new debt price
        itp_q = make_itp(dd, new_q);

        # Update value function 
        vfi_iter!(new_v, itp_q, dd)
        norm_v = norm(dd.v[:V])
        dist_v = norm(new_v - dd.v[:V]) / max(1, norm_v)

        # Distances
        dist = max(dist_q / 500, dist_v)

        # Save all
        dd.v[:V] .= new_v
        dd.q .= dd.q + upd_η * (new_q - dd.q)
        dd.qD .= dd.qD + upd_η * (new_qd - dd.qD)

        verbose && print("dist (v,q) = ($(@sprintf("%0.3g", dist_v)), $(@sprintf("%0.3g", dist_q))) at |v| = $(@sprintf("%0.3g", norm_v)) \n")

        upd_η = max(0.005, upd_η * 0.99)
    end
    if tinyreport
        dist < tol ? print("✓ ($iter) ") : print("($(@sprintf("%0.3g", dist)) after $iter) ")
    else
        dist < tol ? print("Converged ") : print("Got ")
        print("to $(@sprintf("%0.3g", dist)) after $iter iterations.\n")
    end
    return dist < tol
end


function solve_eval_ατ(dd::DebtMod, α, τ)
    dd.pars[:α] = α
    dd.pars[:τ] = τ
    Ny = length(dd.gr[:y])
    mpe!(dd)
    return dd.v[:V][1, ceil(Int, Ny/2)]
end
