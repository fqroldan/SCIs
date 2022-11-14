function update_q_RE!(new_q, new_qd, itp_q, itp_qd, r, dd::Default)
    ρ, ψ, ℏ = (dd.pars[key] for key in (:ρ, :ψ, :ℏ))

    for (jbp, bpv) in enumerate(dd.gr[:b]), (jy, yv) in enumerate(dd.gr[:y])

        Eq = 0.0
        EqD = 0.0

        for (jyp, ypv) in enumerate(dd.gr[:y])
            prob = dd.P[:y][jy, jyp]
            
            prob_def = dd.v[:prob][jbp, jyp]
            bpp = dd.gb[jbp, jyp]

            coupon = coupon_rate(ypv, dd)

            rep_R = (1 - prob_def) * (coupon + (1 - ρ) * itp_q(bpp, jyp)) + prob_def * (1 - ℏ) * itp_qd((1 - ℏ) * bpv, jyp)

            rep_D = ψ * rep_R + (1 - ψ) * dd.qD[jbp, jyp]

            Eq += prob * rep_R
            EqD += prob * rep_D
        end

        new_q[jbp, jy] = Eq  / (1+r)
        new_qd[jbp, jy] = EqD / (1+r)
    end
end

function q_RE(dd::Default, r=dd.pars[:r]; tol=1e-6, maxiter=2_000, verbose=false, do_calc = true)
    dist = 1 + tol
    iter = 0

    if isapprox(dd.pars[:θ], 0)
        do_calc = false
    end

    q_star  = copy(dd.q)
    q_stard = copy(dd.qD)
    new_q   = similar(q_star)
    new_qd  = similar(q_star)

    if do_calc
        while dist > tol && iter < maxiter
            iter += 1

            itp_q  = make_itp(dd, q_star)
            itp_qd = make_itp(dd, q_stard)

            update_q_RE!(new_q, new_qd, itp_q, itp_qd, r, dd)
            dist = norm(new_q - q_star) / max(1, norm(q_star))

            q_star  .= new_q
            q_stard .= new_qd

            verbose && print("Iteration $iter, d = $dist\n")
        end
    end

    itp_qRE = interpolate((dd.gr[:b], dd.gr[:y]), q_star, Gridded(Linear()))
    itp_qdRE = interpolate((dd.gr[:b], dd.gr[:y]), q_stard, Gridded(Linear()))

    return itp_qRE, itp_qdRE
end

function update_q_nodef!(new_q, q_star, r, dd::Default, α, τ)
    ρ = dd.pars[:ρ]

    for jy in eachindex(dd.gr[:y])
        Eq = 0.0
        py = dd.P[:y][jy, :]
        for (jyp, ypv) in enumerate(dd.gr[:y])
            prob = py[jyp]
        
            coupon = coupon_rate(ypv, dd, α = α, τ = τ)
            resale = (1-ρ) * q_star[jyp]

            cond_rep = coupon + resale

            Eq += prob * cond_rep
        end
        new_q[jy] = Eq / (1+r)
    end
end

function q_nodef(dd::Default, r::Float64, α, τ; tol = 1e-6, maxiter = 2_000, verbose = false)
    dist = 1+tol
    iter = 0

    q_star = zeros(length(dd.gr[:y]))
    new_q = similar(q_star)
    
    while dist > tol && iter < maxiter
        iter += 1
        
        update_q_nodef!(new_q, q_star, r, dd, α, τ)
        dist = norm(new_q - q_star) / max(1, norm(q_star))

        q_star .= new_q

        verbose && print("Iteration $iter, d = $dist\n")
    end
    return q_star
end

function yield_grid(dd::Default, Nr, α, τ)
    Ny = length(dd.gr[:y])
    rgrid = range(0.0, 0.2, length=Nr+1)[2:end]

    q_mat = zeros(Nr, Ny)

    Threads.@threads for jr in eachindex(rgrid)
        rv = rgrid[jr]

        qstar = q_nodef(dd, rv, α, τ)

        q_mat[jr, :] .= qstar
    end
    return q_mat, rgrid
end

function yields_from_q(q_mat, rgrid, dd::Default, Nq)
    Ny = length(dd.gr[:y])

    rmin, rmax = extrema(rgrid)
    itp_q = interpolate((rgrid, dd.gr[:y]), q_mat, Gridded(Linear()))
    qgrid = range(extrema(q_mat)..., length=Nq)

    r_mat = zeros(Nq, Ny)

    for (jq, qv) in enumerate(qgrid), (jy, yv) in enumerate(dd.gr[:y])

        f(r) = (itp_q(r, yv) - qv).^2

        res = Optim.optimize(f, rmin, rmax, GoldenSection())

        r_mat[jq, jy] = res.minimizer
    end
    return r_mat, qgrid
end

function get_yields(dd::Default, Nr, Nq, α, τ)
    
    q_mat, rgrid = yield_grid(dd, Nr, α, τ)
    r_mat, qgrid = yields_from_q(q_mat, rgrid, dd, Nq)

    return r_mat, qgrid
end

function get_yields_itp(dd::Default, Nr = 400, Nq = 800; α = dd.pars[:α], τ = dd.pars[:τ])

    r_mat, qgrid = get_yields(dd, Nr, Nq, α, τ)
    itp_yield = interpolate((qgrid, dd.gr[:y]), r_mat, Gridded(Linear()))

    itp_yield = extrapolate(itp_yield, Interpolations.Flat())
    return itp_yield
end

get_spread(q, κ::Number) = κ * (1/q - 1)
get_spread(q, dd::DebtMod) = get_spread(q, dd.pars[:κ])


function q_iter_local!(new_q, new_qd, old_q, old_qD, dd::Default, vL)
    """ Ecuación de Euler de los acreedores determinan el precio de la deuda dada la deuda, el ingreso, y el precio esperado de la deuda """
    ρ, ℏ, ψ, r, θ = (dd.pars[sym] for sym in (:ρ, :ℏ, :ψ, :r, :θ))

    # Interpola el precio de la deuda (para mañana)
    itp_qd = make_itp(dd, old_qD);
    itp_q = make_itp(dd, old_q);

    for (jbp, bpv) in enumerate(dd.gr[:b]), (jy, yv) in enumerate(dd.gr[:y])
        Eq = 0.0
        EqD = 0.0
        sum_sdf_R = 0.0
        sum_sdf_D = 0.0
        for (jyp, ypv) in enumerate(dd.gr[:y])
            prob_def = dd.v[:prob][jbp, jyp]
        
            if θ > 1e-3
                sdf_R = exp(-θ * vL[jbp, jyp, 1])
                sdf_D = exp(-θ * vL[jbp, jyp, 2])
            else
                sdf_R = 1.
                sdf_D = 1.
            end
        
            coupon = coupon_rate(ypv, dd, α = 1, τ = 1)
        
            # Si el país tiene acceso a mercados, emite y puede hacer default mañana
            bpp = dd.gb[jbp, jyp]
            rep_R = (1 - prob_def) * sdf_R * (coupon + (1 - ρ) * itp_q(bpp, jyp)) + prob_def * sdf_D * (1 - ℏ) * itp_qd((1 - ℏ) * bpv, jyp)
        
            # Si el país está en default, mañana puede recuperar acceso a mercados
            rep_D = ψ * rep_R + (1 - ψ) * sdf_D * old_qD[jbp, jyp]
        
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

function marginal_threshold_issue(dd::DebtMod; tol=1e-6, maxiter=500, verbose = false)

    iter = 0
    dist = 1+tol

    q  = copy(dd.q)
    qD = copy(dd.qD)

    new_q  = similar(dd.q)
    new_qD = similar(dd.qD)

    vL = similar(dd.vL)

    while dist > tol && iter < maxiter
        iter += 1

        v_lender_iter!(dd, vL, q, 1, 1)
        q_iter_local!(new_q, new_qD, q, qD, dd, vL)
        
        dist_qR = norm(new_q - q) / max(1, norm(q))
        dist_qD = norm(new_qD - qD)/max(1, norm(qD))
        
        dist = max(dist_qD, dist_qR)

        q  .= new_q
        qD .= new_qD
    end
    verbose && print("Done in $iter iterations")
    return q, qD
end

function q_SDF_og(dd::DebtMod, do_calc=true; tol=1e-6, maxiter=500, verbose = false)
    iter = 0
    dist = 1+tol

    q  = copy(dd.q)
    qD = copy(dd.qD)

    new_q  = similar(dd.q)
    new_qD = similar(dd.qD)

    while dist > tol && iter < maxiter
        iter += 1

        q_iter_local!(new_q, new_qD, q, qD, dd, dd.vL)
        
        dist_qR = norm(new_q - q) / max(1, norm(q))
        dist_qD = norm(new_qD - qD)/max(1, norm(qD))
        
        dist = max(dist_qD, dist_qR)

        q  .= new_q
        qD .= new_qD
    end
    verbose && print("Done in $iter iterations")
    return q, qD
end

function itp_mti(dd::DebtMod; α = 1, τ = 1, do_calc=true)

    spr_og = zeros(length(dd.gr[:b]), length(dd.gr[:y]), 1:2)
    if α == dd.pars[:α] && τ == dd.pars[:τ]
        do_calc = false
    end
       
    q, qD = q_SDF_og(dd, do_calc)
    itp_yield = get_yields_itp(dd::Default, α = α, τ = τ)

    for jb in eachindex(dd.gr[:b]), (jy, yv) in enumerate(dd.gr[:y])
        # 1 = repayment
        # 2 = default

        spr_og[jb, jy, 1] = itp_yield(q[jb, jy], yv)
        spr_og[jb, jy, 2] = itp_yield(qD[jb, jy], yv)
    end


    knts = (dd.gr[:b], dd.gr[:y], 1:2)
    itp_spr_og = interpolate(knts, spr_og, Gridded(Linear()))
end
