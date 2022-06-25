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

# 0.0785 = λ + (1-λ) z = 0.05 + 0.95*0.03 in CE
CE_params(;kwargs...) = Default(γ = 2, σy = 0.027092, ρy = 0.948503, ψ = 0.0385, r = 0.01, ρ = 0.05, β = 0.95402, d1 = -0.18819, d2 = 0.24558, κ = 0.0785; kwargs...)

function Default(;
    β=0.9627,
    γ=2,
    r=0.01,
    θ=1.6155,
    ψ=0.0385,
    χ = 0.0025,
    α=0,
    τ=0,
    ρ=0.05,
    ℏ=1.0,
    d1 = -0.255,
    d2 = 0.296,
    ρy=0.9484,
    σy=0.019,
    # σy = 0.026,
    Nb=200,
    Ny=41,
    bmax=1,
    std_devs = 4,
    κ = r+ρ,
)

    # d1 = 1-Δ
    # d2 = 0

    βL = 1/(1+r)
    wL = 1

    pars = Dict(:β => β, :γ => γ, :r => r, :θ => θ, :χ => χ, :ρ => ρ, :κ => κ, :ℏ => ℏ, :d1 => d1, :d2 => d2, :ρy => ρy, :σy => σy, :α => α, :τ => τ, :ψ => ψ, :βL => βL, :wL => wL)

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

function logsumexp(a::AbstractVector{<:Real})
    m = maximum(a)
    return m + log.(sum(exp.(a .- m)))
end

u(cv, dd::DebtMod) = u(cv, dd.pars[:γ])
function u(cv, γ)
    cmin = 1e-5
    if cv < cmin
        # Por debajo de cmin, lineal con la derivada de u en cmin
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

function coupon_rate(yv, dd::DebtMod)
    α, τ, κ = (dd.pars[sym] for sym in (:α, :τ, :κ))

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
    
    # consumo es ingreso más ingresos por vender deuda nueva menos repago de deuda vieja
    cv = yv + q * (bpv - (1 - ρ) * bv) - coupon * bv
    return cv
end
function eval_value(jb, jy, bpv, itp_q, itp_v, dd::DebtMod)
    """ Evalúa la función de valor en (b,y) para una elección de b' """
    β = dd.pars[:β]
    bv, yv = dd.gr[:b][jb], dd.gr[:y][jy]

    # Interpola el precio de la deuda para el nivel elegido
    qv = itp_q(bpv, yv)

    # Deduce consumo del estado, la elección de deuda nueva y el precio de la deuda nueva
    cv = budget_constraint(bpv, bv, yv, qv, dd)

    # Evalúa la función de utilidad en c
    ut = u(cv, dd)

    # Calcula el valor esperado de la función de valor interpolando en b'
    Ev = 0.0
    for (jyp, ypv) in enumerate(dd.gr[:y])
        prob = dd.P[:y][jy, jyp]
        Ev += prob * itp_v(bpv, ypv)
    end

    # v es el flujo de hoy más el valor de continuación esperado descontado
    v = ut + β * Ev

    return v, cv
end

function opt_value(jb, jy, itp_q, itp_v, dd::DebtMod)
    """ Elige b' en (b,y) para maximizar la función de valor """

    # b' ∈ bgrid
    b_min, b_max = extrema(dd.gr[:b])

    # Función objetivo en términos de b', dada vuelta 
    obj_f(bpv) = -eval_value(jb, jy, bpv, itp_q, itp_v, dd)[1]

    # Resuelve el máximo
    res = Optim.optimize(obj_f, b_min, b_max, GoldenSection())

    # Extrae el argmax
    b_star = res.minimizer

    # Extrae v y c consistentes con b'
    vp, c_star = eval_value(jb, jy, b_star, itp_q, itp_v, dd)

    return vp, c_star, b_star
end

function value_default(jb, jy, dd::DebtMod)
    β, ψ = (dd.pars[sym] for sym in (:β, :ψ))
    """ Calcula el valor de estar en default en el estado (b,y) """
    yv = dd.gr[:y][jy]

    # Consumo en default es el ingreso menos los costos de default
    c = cons_in_default(yv, dd)

    c > 1e-2 || println("WARNING: negative c at (jb, jy) = ($jb, $jy)")

    # Valor de continuación tiene en cuenta la probabilidad ψ de reacceder a mercados
    Ev = 0.0
    for jyp in eachindex(dd.gr[:y])
        prob = dd.P[:y][jy, jyp]
        Ev += prob * (ψ * dd.v[:V][jb, jyp] + (1 - ψ) * dd.v[:D][jb, jyp])
    end

    v = u(c, dd) + β * Ev

    return c, v
end

function vfi_iter!(new_v, itp_q, dd::DebtMod)
    # Reconstruye la interpolación de la función de valor
    knots = (dd.gr[:b], dd.gr[:y])
    itp_v = interpolate(knots, dd.v[:V], Gridded(Linear()))

    Threads.@threads for jb in eachindex(dd.gr[:b])
        for jy in eachindex(dd.gr[:y])

            # En repago
            vp, c_star, b_star = opt_value(jb, jy, itp_q, itp_v, dd)

            # Guarda los valores para repago 
            dd.v[:R][jb, jy] = vp
            dd.gb[jb, jy] = b_star
            dd.gc[jb, jy, 1] = c_star

            # En default
            cD, vD = value_default(jb, jy, dd)
            dd.v[:D][jb, jy] = vD
            dd.gc[jb, jy, 2] = cD
        end
    end

    χ, ℏ = (dd.pars[sym] for sym in (:χ, :ℏ))
    itp_vD = interpolate(knots, dd.v[:D], Gridded(Linear()))
    for (jb, bv) in enumerate(dd.gr[:b]), (jy, yv) in enumerate(dd.gr[:y])
        # Valor de repagar y defaultear llegando a (b,y)
        vr = dd.v[:R][jb, jy]
        vd = itp_vD((1 - ℏ) * bv, yv)

        ## Modo 2: valor extremo tipo X evitando comparar exponenciales de cosas grandes
        lse = logsumexp([vd / χ, vr / χ])
        lpr = vd / χ - lse
        pr = exp(lpr)
        V = χ * lse

        # Guarda el valor y la probabilidad de default al llegar a (b,y)
        new_v[jb, jy] = V
        dd.v[:prob][jb, jy] = pr
    end
end

function value_lenders(bv, bpv, yv, py, itp_q, itp_def, itp_vL, dd::DebtMod; rep)
    θ, wL, βL, ρ, ψ = (dd.pars[sym] for sym in (:θ, :wL, :βL, :ρ, :ψ))
    
    cL = wL
    if rep
        coupon = coupon_rate(yv, dd)
        cL += coupon * bv - itp_q(bpv, yv) * (bpv - (1 - ρ) * bv)
    end

    if !rep
        bpv = bv
    end

    Ev = 0.0
    for (jyp, ypv) in enumerate(dd.gr[:y])
        prob = py[jyp]

        p_def = 1 - ψ
        if rep
            p_def = itp_def(bpv, ypv)
        end

        if θ > 1e-2
            vL_cond = p_def * exp(-θ * itp_vL(bpv, ypv, 2)) + (1 - p_def) * exp(-θ * itp_vL(bpv, ypv, 1))
        else
            vL_cond = p_def * itp_vL(bpv, ypv, 2) + (1-p_def) * itp_vL(bpv, ypv, 2)
        end
        Ev += prob * vL_cond
    end

    if θ > 1e-2
        vL = cL + βL / (-θ) * log(Ev)
    else
        vL = cL + βL * Ev
    end

    return vL
end

function v_lender_iter!(dd::Default)
    knots = (dd.gr[:b], dd.gr[:y])
    itp_q = interpolate(knots, dd.q, Gridded(Linear()))
    itp_def = interpolate(knots, dd.v[:prob], Gridded(Linear()))

    knots = (dd.gr[:b], dd.gr[:y], 1:2)
    itp_vL = interpolate(knots, dd.vL, Gridded(Linear()))

    for (jb, bv) in enumerate(dd.gr[:b]), (jy, yv) in enumerate(dd.gr[:y])
        py = dd.P[:y][jy, :]

        bpv = dd.gb[jb, jy]
        dd.vL[jb, jy, 1] = value_lenders(bv, bpv, yv, py, itp_q, itp_def, itp_vL, dd, rep=true)
        dd.vL[jb, jy, 2] = value_lenders(bv, bpv, yv, py, itp_q, itp_def, itp_vL, dd, rep=false)
    end
end

function q_iter!(new_q, new_qd, dd::Default)
    """ Ecuación de Euler de los acreedores determinan el precio de la deuda dada la deuda, el ingreso, y el precio esperado de la deuda """
    ρ, ℏ, ψ, r, θ = (dd.pars[sym] for sym in (:ρ, :ℏ, :ψ, :r, :θ))

    # Interpola el precio de la deuda (para mañana)
    knots = (dd.gr[:b], dd.gr[:y])
    itp_qd = interpolate(knots, dd.qD, Gridded(Linear()))
    itp_q = interpolate(knots, dd.q, Gridded(Linear()))

    for (jbp, bpv) in enumerate(dd.gr[:b]), (jy, yv) in enumerate(dd.gr[:y])
        Eq = 0.0
        EqD = 0.0
        sum_sdf_R = 0.0
        sum_sdf_D = 0.0
        for (jyp, ypv) in enumerate(dd.gr[:y])
            prob_def = dd.v[:prob][jbp, jyp]
        
            if θ > 1e-2
                sdf_R = exp(-θ * dd.vL[jbp, jyp, 1])
                sdf_D = exp(-θ * dd.vL[jbp, jyp, 2])
            else
                sdf_R = 1
                sdf_D = 1
            end
        
            coupon = coupon_rate(ypv, dd)
        
            # Si el país tiene acceso a mercados, emite y puede hacer default mañana
            bpp = dd.gb[jbp, jyp]
            rep_R = (1 - prob_def) * sdf_R * (coupon + (1 - ρ) * itp_q(bpp, ypv)) + prob_def * sdf_D * (1 - ℏ) * itp_qd((1 - ℏ) * bpv, ypv)
        
            # Si el país está en default, mañana puede recuperar acceso a mercados
            rep_D = ψ * rep_R + (1 - ψ) * sdf_D * dd.qD[jbp, jyp]
        
            prob = dd.P[:y][jy, jyp]
            Eq += prob * rep_R
            EqD += prob * rep_D
        
            sum_sdf_R += prob * (prob_def * sdf_D + (1 - prob_def) * sdf_R)
            sum_sdf_D += prob * ((1-ψ) * sdf_D + ψ * sdf_R)
        end
        new_q[jbp, jy] = Eq / (1 + r) / sum_sdf_R
        new_qd[jbp, jy] = EqD / (1 + r) / sum_sdf_D
    end
end

function mpe!(dd::Default; tol=1e-6, maxiter=500, min_iter = 1, verbose = true)

    new_v = similar(dd.v[:V])
    new_q = similar(dd.q)
    new_qd = similar(dd.qD)

    dist = 1 + tol
    iter = 0

    knots = (dd.gr[:b], dd.gr[:y])

    while iter < min_iter || (dist > tol && iter < maxiter)
        iter += 1

        verbose && print("Iteration $iter: ")

        # Actualiza el precio de la deuda
        v_lender_iter!(dd)
        q_iter!(new_q, new_qd, dd)
        dist_qR = norm(new_q - dd.q) / max(1, norm(dd.q))
        dist_qD = norm(new_qd - dd.qD)/max(1, norm(dd.qD))
        dist_q = max(dist_qD, dist_qR)

        # Interpolación del precio de la deuda
        itp_q = interpolate(knots, new_q, Gridded(Linear()))

        # Actualiza la función de valor
        vfi_iter!(new_v, itp_q, dd)
        norm_v = norm(dd.v[:V])
        dist_v = norm(new_v - dd.v[:V]) / max(1, norm_v)

        # Distancias
        dist = max(dist_q / 500, dist_v)

        # Guardamos todo
        dd.v[:V] .= new_v
        dd.q .= new_q
        dd.qD .= new_qd

        verbose && print("dist (v,q) = ($(@sprintf("%0.3g", dist_v)), $(@sprintf("%0.3g", dist_q))) at |v| = $(@sprintf("%0.3g", norm_v)) \n")
    end
    dist < tol && print("Converged to $(@sprintf("%0.3g", dist)) after $iter iterations.\n")
    
    nothing
end

# @time Optim.optimize(α -> -solve_eval_α(dd, α), -0.1, 8, GoldenSection())

function solve_eval_ατ(dd::DebtMod, α, τ)
    dd.pars[:α] = α
    dd.pars[:τ] = τ
    Ny = length(dd.gr[:y])
    mpe!(dd)
    return dd.v[:V][1, ceil(Int, Ny/2)]
end

# Optim.optimize(x -> -solve_eval_ατ(dd, x[1], x[2]), [-0.1, minimum(dd.gr[:y])], [10, dd.gr[:y][floor(Int, length(dd.gr[:y])*0.6)]], [0, 0.0], Fminbox(NelderMead()))