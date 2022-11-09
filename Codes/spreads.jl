function update_q_nodef!(new_q, q_star, r, dd::Default)
    ρ = dd.pars[:ρ]

    for jy in eachindex(dd.gr[:y])
        Eq = 0.0
        py = dd.P[:y][jy, :]
        for (jyp, ypv) in enumerate(dd.gr[:y])
            prob = py[jyp]
        
            coupon = coupon_rate(ypv, dd)
            resale = (1-ρ) * q_star[jyp]

            cond_rep = coupon + resale

            Eq += prob * cond_rep
        end
        new_q[jy] = Eq / (1+r)
    end
end

function q_nodef(dd::Default, r::Float64; tol = 1e-6, maxiter = 2_000, verbose = false)
    dist = 1+tol
    iter = 0

    q_star = zeros(length(dd.gr[:y]))
    new_q = similar(q_star)
    
    while dist > tol && iter < maxiter
        iter += 1
        
        update_q_nodef!(new_q, q_star, r, dd)
        dist = norm(new_q - q_star) / max(1, norm(q_star))

        q_star .= new_q

        verbose && print("Iteration $iter, d = $dist\n")
    end
    return q_star
end

function yield_grid(dd::Default, Nr)
    Ny = length(dd.gr[:y])
    rgrid = range(0.0, 0.2, length=Nr+1)[2:end]

    q_mat = zeros(Nr, Ny)

    Threads.@threads for jr in eachindex(rgrid)
        rv = rgrid[jr]

        qstar = q_nodef(dd, rv)

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

function get_yields(dd::Default, Nr, Nq)
    
    q_mat, rgrid = yield_grid(dd, Nr)
    r_mat, qgrid = yields_from_q(q_mat, rgrid, dd, Nq)

    return r_mat, qgrid
end

function get_yields_itp(dd::Default, Nr = 400, Nq = 800)

    r_mat, qgrid = get_yields(dd, Nr, Nq)
    itp_yield = interpolate((qgrid, dd.gr[:y]), r_mat, Gridded(Linear()))

    itp_yield = extrapolate(itp_yield, Interpolations.Flat())
    return itp_yield
end

get_spread(q, κ::Number) = κ * (1/q - 1)
get_spread(q, dd::DebtMod) = get_spread(q, dd.pars[:κ])


function spread_decomp(dd::DebtMod)
    ρ, ℏ, θ, βL = (dd.pars[sym] for sym in (:ρ, :ℏ, :ψ, :r, :θ, :βL))

    # Interpola el precio de la deuda (para mañana)
    itp_qd = make_itp(dd, dd.qD);
    itp_q = make_itp(dd, dd.q);

    q_RE = similar(dd.q)
    qθ_def = similar(dd.q)
    qθ_cont = similar(dd.q)

    for (jbp, bpv) in enumerate(dd.gr[:b]), (jy, yv) in enumerate(dd.gr[:y])
        Eq_RE = 0.0
        E_P     = 0.0
        E_PD    = 0.0
        E_M     = 0.0
        E_d     = 0.0
        E_Md    = 0.0
        E_MP    = 0.0
        E_MPD   = 0.0
        
        sum_sdf = 0.0
        
        for (jyp, ypv) in enumerate(dd.gr[:y])
            prob = dd.P[:y][jy, jyp]

            prob_def = dd.v[:prob][jbp, jyp]
        
            if θ > 1e-3
                sdf_R = exp(-θ * dd.vL[jbp, jyp, 1])
                sdf_D = exp(-θ * dd.vL[jbp, jyp, 2])
            else
                sdf_R = 1.
                sdf_D = 1.
            end
        
            coupon = coupon_rate(ypv, dd)
        
            # Si el país tiene acceso a mercados, emite y puede hacer default mañana
            bpp = dd.gb[jbp, jyp]

            P  = coupon + (1-ρ) * itp_q(bpp, jyp)
            PD = (1-ℏ) * itp_qd((1-ℏ) * bpv, jyp)

            Eq_RE += prob * ((1 - prob_def) * P + (prob_def) * PD)

            E_P   += prob * P

            E_PD  += prob * PD

            sum_sdf += prob * (prob_def * sdf_D + (1-prob_def) * sdf_R)
            E_M   += prob * βL * (prob_def * sdf_D + (1-prob_def) * sdf_R)

            E_d   += prob * prob_def * 1

            E_Md  += prob * βL * (prob_def * sdf_D + (1-prob_def) * 0)

            E_MP  += prob * P * βL * (prob_def * sdf_D + (1-prob_def) * sdf_R)
            E_MPD += prob * PD* βL * (prob_def * sdf_D + (1-prob_def) * sdf_R)
        end

        cov_M_d  = (E_Md - E_M * E_d) / sum_sdf
        cov_M_P  = (E_MP - E_M * E_P) / sum_sdf
        cov_M_PD = (E_MPD- E_M * E_PD)/ sum_sdf
        
        q_RE[jbp, jy] = E_M / sum_sdf * Eq_RE

        qθ_def[jbp, jy] = - (E_P - E_PD) * cov_M_d

        qθ_cont[jbp, jy] = (1-E_d) * cov_M_P + E_d * cov_M_PD
    end

    return q_RE, qθ_def, qθ_cont
end