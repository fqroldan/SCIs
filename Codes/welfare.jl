function simul_cons(dd::DebtMod, T=150, K=10_000)
    
    ϵvv, ξvv = simulshocks(T, K);

    itp_yield = get_yields_itp(dd);
    itp_qRE, itp_qdRE = q_RE(dd, do_calc = false);
    itp_spr_og = itp_mti(dd, do_calc = false);

    Ny = length(dd.gr[:y])
    c_bar = Vector{Vector{Float64}}(undef, Ny)
    u_bar = Vector{Vector{Float64}}(undef, Ny)

    for (jy, yv) in enumerate(dd.gr[:y])
        pv, _ = simulvec(dd, itp_yield, itp_qRE, itp_qdRE, itp_spr_og, ϵvv, ξvv, burn_in = 0, stopdef = false, B0 = 0., Y0 = yv)

        c_all = [ifelse(pp[:ζ, t] == 2, pp[:y, t], pp[:c, t]) for t in 1:length(pv[1][:ζ]), pp in pv]

        u_bar[jy] = mean([u(c, dd) for c in c_all], dims=2) |> vec
        c_bar[jy] = mean(c_all, dims = 2) |> vec

    end
    return u_bar, c_bar
end

function NPDV(dd::DebtMod, yvec)
    β = dd.pars[:β]

    T = length(yvec)

    v = 0.0
    for jt in 1:T
        v += β^jt * yvec[jt]
    end

    return v
end

function welfare_decomp(dd::DebtMod; T = 150, K = 10_000)

    u_bar, c_bar = simul_cons(dd, T, K)
    u_cbar = [u(c, dd) for c in c_bar]

    w = dd.v[:V][1, :]

    w_nocost = [NPDV(dd, u_cbar[jy]) for jy in eachindex(dd.gr[:y])]
    w_u_bar  = [NPDV(dd, u_bar[jy])  for jy in eachindex(dd.gr[:y])]

    return w, w_nocost, w_u_bar
end


### Stuff below unused


function vfic!(dd::DebtMod; tol = 1e-6, maxiter = 2_000)

    dist = 1+tol
    iter = 0
    
    vR = similar(dd.v[:R])
    vD = similar(dd.v[:D])

    new_v = similar(dd.v[:V])
    old_v = copy(dd.v[:V])

    itp_q = make_itp(dd, dd.q)

    while dist > tol && iter < maxiter

        iter += 1

        itp_v = make_itp(dd, old_v)
        vfic_iter!(new_v, vR, vD, dd, itp_q, itp_v)

        dist = norm(new_v - old_v) / max(1, norm(old_v))

        old_v .= new_v
    end
end

function value_default(jb, jy, dd::DebtMod, itp_Ev, Evd)
    β, ψ = (dd.pars[sym] for sym in (:β, :ψ))
    """ Calcula el valor de estar en default en el estado (b,y) """
    bv, yv = dd.gr[:b][jb], dd.gr[:y][jy]

    Ev = ψ * itp_Ev(bv) + (1-ψ) * Evd[jb]

    v = u(yv, dd) + β * Ev

    return v
end

function vfic_iter!(new_v, vR, vD, dd::DebtMod, itp_q, old_v)
    χ, ℏ = (dd.pars[key] for key in (:χ, :ℏ))
    Threads.@threads for jy in eachindex(dd.gr[:y])
        
        itp_Ev = make_itp_vp(dd, jy, old_v);
        Evd    = make_vp(dd, jy, vD)

        for jb in eachindex(dd.gr[:b])
            bp = dd.gb[jb, jy]
            vR[jb, jy] = eval_value(jb, jy, bp, itp_q, itp_Ev, dd)

            vD[jb, jy] = value_default(jb, jy, dd, itp_Ev, Evd)
        end

        itp_vd = make_itp_vp(dd, jy, vD)
        vs = zeros(2)
        for (jb, bv) in enumerate(dd.gr[:b])
            vr = vR[jb, jy]
            vd = itp_vd((1-ℏ)*bv)

            vs .= (vd, vr)

            lse = logsumexp(Vs ./ χ)
            # lpr = vd / χ - lse
            # pr = exp(lpr)
            V = χ * lse

            new_v[jb, jy] = V
        end
    end
end
