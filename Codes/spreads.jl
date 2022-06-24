function update_q_nodef!(new_q, q_star, r, dd::Default)
    ρ = dd.pars[:ρ]

    for (jy, yv) in enumerate(dd.gr[:y])
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
        dist = norm(new_q - q_star) / (1+norm(q_star))

        q_star .= new_q

        verbose && print("Iteration $iter, d = $dist\n")
    end
    return q_star
end

function yield_grid(dd::Default, Nr = 801)
    Ny = length(dd.gr[:y])
    rgrid = range(0.01, 0.2, length=Nr)

    q_mat = zeros(Nr, Ny)

    for (jr, rv) in enumerate(rgrid)

        qstar = q_nodef(dd, rv)

        q_mat[jr, :] .= qstar
    end
    return q_mat, rgrid
end

function yields_from_q(q_mat, rgrid, dd::Default, Nq = 801)
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

function get_yields(dd::Default, Nr = 801, Nq = Nr)
    
    q_mat, rgrid = yield_grid(dd, Nr)
    r_mat, qgrid = yields_from_q(q_mat, rgrid, dd, Nq)

    return r_mat, qgrid
end

function get_yields_itp(dd::Default, Nr = 801, Nq = Nr)

    r_mat, qgrid = get_yields(dd, Nr, Nq)
    itp_yield = interpolate((qgrid, dd.gr[:y]), r_mat, Gridded(Linear()))

    itp_yield = extrapolate(itp_yield, Interpolations.Flat())
    return itp_yield
end

get_spread(q, κ::Number) = κ * (1/q - 1)
get_spread(q, dd::DebtMod) = get_spread(q, dd.pars[:κ])