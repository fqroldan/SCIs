using QuantEcon, Distributions, PlotlyJS, Optim, Interpolations, ForwardDiff, ColorSchemes, Printf, CSV, DataFrames, NLopt

sand() = "#F5F3F1"
darkbgd() = "#272929"
lightgrid() = "#353535"
darkgrid() = "#e2e2e2"
gridcol(dark=false) = ifelse(dark, lightgrid(), darkgrid())

q_axis(dark) = attr(showgrid = true, gridcolor=gridcol(dark), gridwidth = 0.5, zeroline=false)
bgcol(slides, dark) = ifelse(slides, ifelse(dark, darkbgd(), sand()), "white")
qleg() = attr(orientation = "h", x=0.05, xanchor="left")

qwidth(slides) = 864
qheight(slides) = ceil(Int, qwidth(slides) * ifelse(slides, 10/16, 7/16))

function qtemplate(;dark=false, slides=!dark)
    axis = q_axis(dark)
    width = 864 #1920 * 0.45
    l = Layout(
        xaxis = axis, yaxis = axis,
        width = width,
        height = width * ifelse(slides, 10/16, 7/16),
        font = attr(
            family = ifelse(slides, "Lato", "Linux Libertine"),
            size = 16, color = ifelse(dark, sand(), darkbgd())
        ),
        paper_bgcolor = bgcol(slides, dark), plot_bgcolor = bgcol(slides, dark),
        legend = qleg(),
    )
    return Template(layout = l)
end

function write_tab2(rb, rf, γ, Δ, g)
	params = [
		"\$\\beta_b\$",
		"\$\\beta\$",
		"\$\\gamma\$",
		"\$d_1\$",
		"\$g\$",
		"\$\\tau\$",
		"\$\\sigma_z\$"
	]
	names = [
		"Borrower's discount rate",
		"Risk-free rate",
		"Borrower's risk aversion",
		"Output cost of default",
		"Expected growth rate",
		"Threshold for repayment",
		"Std.~deviation of log output"
	]
	values = [
		"$rb\\% ann.",
		"$rf\\% ann.",
		"$γ",
		"$(100*Δ)\\%",
		"$g\\% ann.",
		"1",
		"0.15"
	]
	parpad = 20
	namepad = maximum(length(name) for name in names) + 2
	valpad = 20

	tab = ""

	for jp in eachindex(params)
		tab *= rpad(params[jp], parpad, " ") * "& "
		tab *= rpad(names[jp], namepad, " ") * "& "
		tab *= rpad(values[jp], valpad, " ") * "\\\\\n"
	end
	return tab
end


mutable struct SCI{K}
	pars::Dict{Symbol, Float64}
	gr::Dict{Symbol, Vector{Float64}}
	Jgr::Array{Int64,K}
	prob::Dict{Symbol, Vector{Float64}}
end

function SCI(;
	βb = (1.03)^-5,
	β = βb,
	γ = 1,
	θ = 2,
	Δ = 0.1,
	σz = 0.15,
	Nz = 71,
	Nξ = 1,
	w1 = 1,
	w2 = 1,
	y1 = 0.5,
	UCP = 0.1,
	ϕ = 0,
	)

	pars = Dict(:β=>β, :βb=>βb, :γ=>γ, :θ=>θ, :Δ=>Δ, :w1=>w1, :w2=>w2, :y1=>y1, :ϕ=>ϕ)

	μz = 1
	zgrid = range(μz-3*σz, μz+3*σz, length=Nz)
	dz = Normal(μz,σz)

	# zgrid = range(μz - 0.5*μz, μz + 0.5*μz, length=Nz)
	# dz = Uniform(extrema(zgrid)...)
	pdf_z = pdf.(dz, zgrid)
	pdf_z /= sum(pdf_z)

	ξgrid = collect(range(-0., 0., length=Nξ))
	pdf_ξ = ones(Nξ) / Nξ

	# push!(ξgrid, 5)
	# pdf_ξ /= sum(pdf_ξ) / (1-3/100)
	# push!(pdf_ξ, 3/100)

	gr = Dict(:z=>zgrid, :ξ=>ξgrid)
	prob = Dict(:z=>pdf_z, :ξ=>pdf_ξ)

	Jgr = gridmake(1:Nz, 1:length(ξgrid))
	K = ndims(Jgr)

	return SCI{K}(pars, gr, Jgr, prob)
end

utility(c::Number, sc::SCI) = utility(c, sc.pars[:γ])
function utility(c::Number, γ::Number)
	cmin = 1e-8
	if c < cmin
		return utility(cmin,γ) + (c-cmin) * (cmin)^-γ
	else
		γ == 1 && return log(c)
		return c^(1-γ)/(1-γ)
	end
end
deriv_utility(c, sc::SCI) = ForwardDiff.derivative(x->utility(x, sc), c)

utility_lenders(sc, cL) = cL # Risk-neutral
deriv_utility_lenders(sc, cL) = ForwardDiff.derivative(x->utility_lenders(sc, x), cL)

y(zv) = zv

budget_constraint_lenders_2(sc::SCI, dv, Rv, bv) = budget_constraint_lenders_2(sc.pars[:w2], dv, Rv, bv)
budget_constraint_lenders_2(w2::Float64, dv, Rv, bv) = w2 + (1-dv)*Rv*bv

function price_otherdebt(sc::SCI, bv, R, R_new)
	Em 		= 0.0
	Er_new 	= 0.0
	def_prob= 0.0
	ER 		= 0.0
	ERd 	= 0.0
	EmR 	= 0.0
	Emd 	= 0.0
	for jj in 1:size(sc.Jgr, 1)
		jz, jξ = sc.Jgr[jj, :]
		prob = sc.prob[:z][jz] * sc.prob[:ξ][jξ]

		zv = sc.gr[:z][jz]
		ξv = sc.gr[:ξ][jξ]

		Rv = R[jz]

		u_def, u_rep = ut_in_def(zv, bv, Rv, sc, 0.0)
		dv = (u_def + ξv > u_rep) #&& Rv > 0
		def_prob += prob * dv

		dv ? u2B = ut_in_def(zv, bv, Rv, sc).d : u2B = ut_in_def(zv, bv, Rv, sc).r

		c2L = budget_constraint_lenders_2(sc.pars[:w2], dv, Rv, bv)
		u2L = utility_lenders(sc, c2L)

		up2L = deriv_utility_lenders(sc, c2L)

		Mv = exp(-sc.pars[:θ] * u2L)
		def_prob += prob * dv

		Rv = R_new[jz]

		Er_new += prob * Mv * Rv * (1-dv)
		Em += prob * Mv

		ER += prob * Rv
		ERd += prob * Rv * (1-dv)
		EmR += prob * Mv * Rv
		Emd += prob * (1-dv) * Mv
	end

	cov_MR = sc.pars[:β] / Em * (EmR - Em*ER)
	cov_dM = sc.pars[:β] / Em * (Emd - Em*(1-def_prob))

	q = sc.pars[:β] * Er_new / Em
	qθ_cont = (1-def_prob) * cov_MR
	qθ_def  = ER * cov_dM
	q_RE    = ERd * sc.pars[:β]

	return q, qθ_cont, qθ_def, q_RE
end

function ut_in_def(zv, bv, Rv, sc::SCI, χc=0)
	yv = y(zv) * (1+χc)

	u_def = utility(yv*(1-sc.pars[:Δ]*sc.pars[:ϕ]) - yv^2*sc.pars[:Δ]*(1-sc.pars[:ϕ]), sc)
	u_rep = utility(yv - Rv*bv, sc)

	(d=u_def, r=u_rep)
end

function def_debtprice!(sc::SCI, bv, R, χc = 0.0)
	Em = 0.0
	Ev = 0.0
	Eu = 0.0
	def_prob = 0.0
	dist_prob = 0.0
	EmR = 0.0
	Emd = 0.0
	ERd = 0.0
	ERb = 0.0
	densvec = zeros(length(sc.gr[:z]))
	l_ratio = zeros(length(sc.gr[:z]))
	def_vec = zeros(length(sc.gr[:z]))
	for jj in 1:size(sc.Jgr, 1)
		jz, jξ = sc.Jgr[jj, :]
		prob = sc.prob[:z][jz] * sc.prob[:ξ][jξ]

		zv = sc.gr[:z][jz]
		ξv = sc.gr[:ξ][jξ]

		Rv = R[jz]

		u_def, u_rep = ut_in_def(zv, bv, Rv, sc, 0.0)
		dv = (u_def + ξv > u_rep) #&& Rv > 0
		def_prob += prob * dv

		dv ? u2B = ut_in_def(zv, bv, Rv, sc, χc).d : u2B = ut_in_def(zv, bv, Rv, sc, χc).r

		c2L = budget_constraint_lenders_2(sc.pars[:w2], dv, Rv, bv)
		u2L = utility_lenders(sc, c2L)

		up2L = deriv_utility_lenders(sc, c2L)

		Mv = exp(-sc.pars[:θ] * u2L)
		
		EmR += prob * Mv * Rv
		Emd += prob * (1-dv) * Mv

		ERd += prob * Rv * (1-dv)

		Em += prob * Mv
		Ev += prob * Mv * (1-dv) * Rv #* up2L

		Eu += prob * u2B
		ERb += prob * Rv * bv

		dist_prob += prob * Mv * dv
		densvec[jz] += prob * Mv
		l_ratio[jz] += sc.prob[:ξ][jξ] * Mv
		def_vec[jz] += dv * sc.prob[:ξ][jξ]
	end
	RHS = sc.pars[:β] * Ev / Em

	dist_prob *= 1/Em
	densvec *= 1/Em
	l_ratio *= 1/Em

	ER = ERb / bv

	cov_MR = sc.pars[:β] / Em * (EmR - Em*ER)
	cov_dM = sc.pars[:β] / Em * (Emd - Em*(1-def_prob))

	qθ_cont = (1-def_prob) * cov_MR
	qθ_def  = ER * cov_dM
	q_RE    = ERd * sc.pars[:β]

	return RHS, Eu, def_prob, ERb, dist_prob, densvec, def_vec, qθ_cont, qθ_def, q_RE, l_ratio
end

function debtprice_RN!(sc::SCI, bv, R, χc = 0.0, χq = 0.0)
	RHS, Eu, def_prob, ER, dist_prob, densvec, def_vec, qθ_cont, qθ_def, q_RE, l_ratio = def_debtprice!(sc, bv, R, χc)

	return RHS, Eu, def_prob, ER, dist_prob, densvec, def_vec, qθ_cont, qθ_def, q_RE, l_ratio
end

function value_b(sc::SCI, bv, R; χc = 0.0, χq = 0.0)
	q1, Eu, def_prob, ER, dist_prob, densvec, def_vec, qθ_cont, qθ_def, q_RE, l_ratio = debtprice_RN!(sc, bv, R, χc, χq)

	c1b = sc.pars[:y1] + q1 * (1-χq) * bv

	v = utility(c1b * (1+χc), sc) + sc.pars[:βb] * Eu
	return v, q1, def_prob, ER, dist_prob, densvec, def_vec, qθ_cont, qθ_def, q_RE, l_ratio
end

function find_maxb(sc::SCI, R, πmax)
	
	function defprob(sc::SCI, bv, R, πmax)
		πv = value_b(sc, first(bv), R)[3]

		thres = 0.5*(1+πmax)
		ifelse(πv >= thres, πv+bv*(πv-thres), πv)
	end

	obj(b) = (defprob(sc, b, R, πmax) - πmax)^2

	bguess = sc.pars[:y1] * 0.99
	res = Optim.optimize(obj, 0.0, bguess)

	res.minimizer
end

function optim_debt_back(sc::SCI, R, maxb = 1.5*sc.pars[:Δ])

	bgrid = range(0, maxb, step=1e-4)

	vtemp = -Inf
	b_opt, v_b, q1, def_prob, ER, qθ_cont, qθ_def, q_RE = zeros(8)
	for (jb, bv) in enumerate(bgrid)
		v_b, q1_b, d_b, ER_b, dist_prob, densvec, def_vec, qθ_cont_b, qθ_def_b, q_RE_b, l_ratio = value_b(sc, bv, R)
		if v_b > vtemp
			vtemp = v_b

			b_opt, q1, def_prob, ER, qθ_cont, qθ_def, q_RE = bv, q1_b, d_b, ER_b, qθ_cont_b, qθ_def_b, q_RE_b
		end
	end

	return b_opt, v_b, q1, def_prob, ER, qθ_cont, qθ_def, q_RE
end


function optim_debt(sc::SCI, R, maxb=3.5*sc.pars[:Δ])

	obj_f(bv) = -value_b(sc, first(bv), R)[1]

	res = Optim.optimize(obj_f, 0, maxb, abs_tol=1e-32, rel_tol=1e-32)
	b_opt = first(res.minimizer)

	v = -res.minimum
	_, q1, def_prob, ER, dist_prob, densvec, def_vec, qθ_cont, qθ_def, q_RE, l_ratio = value_b(sc, b_opt, R)

	# b_opt, v, q1, def_prob, ER, qθ_cont, qθ_def, q_RE = optim_debt_back(sc, R, maxb)

	if isapprox(b_opt, maxb)
		return optim_debt(sc, R, 2*maxb)
	end

	return b_opt, v, q1, def_prob, ER, qθ_cont, qθ_def, q_RE, l_ratio
end

function plots_R(sc::SCI)

	αvec = range(0,1, length=25)
	
	dd = Dict(key => similar(αvec) for key in [:b, :v, :q, :π, :ER])
	
	for (jα, αv) in enumerate(αvec)
		R = 1 .+ αv * (sc.gr[:z].-1)

		b_opt, v, q1, def_prob, ER, qθ_cont, qθ_def, q_RE, l_ratio = optim_debt(sc, R)

		dd[:b][jα] = b_opt
		dd[:v][jα] = v
		dd[:q][jα] = q1
		dd[:π][jα] = def_prob
		dd[:ER][jα] = ER
		# println(q1 - (1-def_prob))
	end

	plot([scatter(x=αvec, y=dd[key], name=key) for key in keys(dd)], Layout(title="θ = $(sc.pars[:θ])"))
end

function Euler_autarky(sc::SCI, bv, R)

	du1 = deriv_utility(sc.pars[:y1] + bv/R, sc)

	du2 = 0.0
	for (jz, zv) in enumerate(sc.gr[:z])
		du2 += sc.prob[:z][jz] * deriv_utility(y(zv)-bv, sc)
	end

	return du1 - sc.pars[:βb] * R * du2
end

function CM_χ(sc::SCI, χ, R_comp, b_comp)

	# NPV_y = sc.pars[:y1] + sc.pars[:β] * 1
	# res = Optim.optimize( bv -> (deriv_utility(sc.pars[:y1] + bv/R, sc) - sc.pars[:βb]*R*deriv_utility(1-bv, sc))^2, 0, 1)
	# b = res.minimizer
	# u2 = utility((1-b)*χ, sc)

	# R = 1/sc.pars[:β]

	# b = 0.0
	# u2 = sum([utility((y(zv)-b)*χ, sc)*sc.prob[:z][jz] for (jz, zv) in enumerate(sc.gr[:z])])

	# c1 = sc.pars[:y1] + b/R

	# v = utility(c1 * χ, sc) + sc.pars[:βb] * u2


	v = value_b(sc, b_comp, R_comp; χc = χ, χq = 0.0)[1]
end


function cons_equiv(sc::SCI, v, R_comp, b_comp)
	res = Optim.optimize(χ -> (CM_χ(sc, χ, R_comp, b_comp) - v)^2, -0.5, 1.5)

	χ = 100*res.minimizer
end

function q_equiv(sc::SCI, v, R_comp, b_comp, q)
	res = Optim.optimize(χ -> (value_b(sc, b_comp, R_comp; χq = χ)[1] - v)^2, -0.5, 1.5)
	χ = res.minimizer

	spread_diff = 10000 * χ/(q*(1-χ))# - 1/q
end

function arrow_securities(sc::SCI)

	obj_f(R) = -value_b(sc, 1, R)[1]

	min_R = ones(length(sc.gr[:z])) * 0
	max_R = sc.gr[:z] * sc.pars[:Δ] * 1.5
	guess_R = sc.gr[:z] * sc.pars[:Δ] * 0.5

	# res = Optim.optimize(obj_f, min_R, max_R, guess_R, Fminbox())

	opt = Opt(:LN_SBPLX, length(guess_R))
	opt.lower_bounds = min_R
	opt.upper_bounds = max_R
	# opt.xtol_rel = 1e-16
	

	F(R,g) = -obj_f(R)
	opt.max_objective = F

	maxf, maxx, ret = NLopt.optimize(opt, guess_R)

	def_prob_max = value_b(sc, 1.001, maxx)[7]

	maxf, maxx, ret, def_prob_max

	# (v = -res.minimum, R = res.minimizer, res = res)
end


function get_spreads(sc, R, q1, qθ_cont, qθ_def, q_RE; PL = 5)
	ER = sc.prob[:z]'*R
	rf = sc.pars[:β]^(-1/PL) - 1
	rd = 1/q_RE^(1/PL) - 1
	r = (ER / q_RE)^(1/PL) - 1

	# r = (1+r)^(1/PL) - 1
	# rd = (1+rd)^(1/PL) - 1

	total_spread = (ER/q1)^(1/PL) - 1 - rf

	sprRE = r - rf
	sprθC = (ER / (q_RE + qθ_cont))^(1/PL) - (ER/q_RE)^(1/PL)
	sprθD = (ER / q1)^(1/PL) - (ER / (q_RE + qθ_cont))^(1/PL)
	
	spread_def = rd - rf

	return sprRE, sprθC, sprθD, spread_def, total_spread
end


function get_spread(sc::SCI, R, q; PL = 5)
	r = 1/q - 1
	rd = sc.prob[:z]'*R / q - 1

	r = (1+r)^(1/PL) - 1
	rd = (1+rd)^(1/PL) - 1

	rf = sc.pars[:β]^(-1/PL) - 1

	spread = r - rf
	spread_def = rd - rf

	return spread, spread_def
end

function plots_alpha(sym = :α; γ=1, y1=0.75, Nα = 25, Nb = 500, Nθ = 5, NΔ = 5, UCP = 1e-8, Nz = 71, template::Template=templates.default, rel_0 = false, show_debt=false, show_q = false)

	αvec = range(0, 1., length = Nα)
	θvec = max.(1e-6, range(0, 4, length=Nθ))
	Δvec = range(0.05, 0.25, length=NΔ)
	# Δvec = range(0.1, length=1)
	NΔ = length(Δvec)

	π_mat = zeros(Nα, Nθ, NΔ)
	v_mat = zeros(Nα, Nθ, NΔ)
	q_mat = zeros(Nα, Nθ, NΔ)
	CEmat = zeros(Nα, Nθ, NΔ)
	CE_FI = zeros(Nθ, NΔ)
	ERmat = zeros(Nα, Nθ, NΔ)
	ER_FI = zeros(Nθ, NΔ)
	qEmat = zeros(Nα, Nθ, NΔ)
	qE_FI = zeros(Nθ, NΔ)

	Δvec = [0.1]

	R_comp = ones(Nz)
	b_comp = 0.0
	v_comp = 0.0
	q_comp = 0.0
	for (jθ, θv) in enumerate(θvec), (jΔ, Δv) in enumerate(Δvec)
		sc = SCI(θ=θv, γ=γ, y1=y1, Δ=Δv, UCP = UCP, Nz = Nz)
		for (jα, αv) in enumerate(αvec)
			bgrid = range(0, 2*Δv, length = Nb)

			if sym == :α
				R = 1 .+ αv * (sc.gr[:z].-1)
			elseif sym == :k
				R = sc.gr[:z] .>= (minimum(sc.gr[:z]) + αv * (maximum(sc.gr[:z]) - minimum(sc.gr[:z])))
			end

			rel_0 && jα == 1 ? R_comp[:] = R : nothing

			vtemp = -Inf
			for (jb, bv) in enumerate(bgrid)
				v, q1, def_prob, ER, dist_prob, densvec, def_vec, qθ_cont, qθ_def, q_RE, l_ratio = value_b(sc, bv, R)
				if v > vtemp
					vtemp = v

					rel_0 && jα == 1 ? b_comp = bv : nothing
					rel_0 && jα == 1 ? v_comp = v  : nothing
					rel_0 && jα == 1 ? q_comp = q1 : nothing

					π_mat[jα, jθ, jΔ] = def_prob
					v_mat[jα, jθ, jΔ] = v
					q_mat[jα, jθ, jΔ] = q1
					ERmat[jα, jθ, jΔ] = ER
					CEmat[jα, jθ, jΔ] = cons_equiv(sc, v, R_comp, b_comp)
					if show_q
						CEmat[jα, jθ, jΔ] = q_equiv(sc, v_comp, R, bv, q1)
					end
				end
			end

		end
		vFI, RFI, _ = arrow_securities(sc)
		q1, _, ER_FI[jθ, jΔ] = value_b(sc, 1, RFI)[2:4]
		CE_FI[jθ, jΔ] = cons_equiv(sc, vFI, R_comp, b_comp)
		if show_q
			CE_FI[jθ, jΔ] = q_equiv(sc, v_comp, RFI, 1, q1)
		end
		# CE_FI[jθ, jΔ] = vFI
		# println("vFI = $vFI")
	end

	Cmax = maximum(CEmat)
	C_α0 = maximum(CEmat[1, :, :])

	gains = Cmax - C_α0

	if rel_0
		title = "Gains from <i>$(sym)=0</i> (2-period)"
	else
		title = "Gains from autarky (2-period)"
	end
	ytitle = "% equiv consumption"
	if show_debt
		title = "Face value of debt"
		ytitle = ""
	end
	if show_q
		ytitle = "Comp inc in spreads (bps)"
	end

	scats = [
		[scatter(x=αvec, y=CEmat[:, jθ, jΔ], legendgroup=jθ, marker_color=get(ColorSchemes.southwest, (jθ-1)/(Nθ-1)), name = "θ = $(@sprintf("%0.3g",θv))") for (jθ, θv) in enumerate(θvec), (jΔ, Δv) in enumerate(Δvec)][:]
		[scatter(x=αvec, y=ones(length(αvec))*CE_FI[jθ,jΔ], showlegend=false, legendgroup=jθ, marker_color=get(ColorSchemes.southwest, (jθ-1)/(Nθ-1)), line_dash="dash",name = "θ=$(@sprintf("%0.3g",θv))") for (jθ, θv) in enumerate(θvec), (jΔ, Δv) in enumerate(Δvec)][:]
		]
	if show_debt
		scats = [
			[scatter(x=αvec, y=ERmat[:, jθ, jΔ], legendgroup=jθ, marker_color=get(ColorSchemes.southwest, (jθ-1)/(Nθ-1)), name = "θ = $(@sprintf("%0.3g",θv))") for (jθ, θv) in enumerate(θvec), (jΔ, Δv) in enumerate(Δvec)][:]
			[scatter(x=αvec, y=ones(length(αvec))*ER_FI[jθ,jΔ], showlegend=false, legendgroup=jθ, marker_color=get(ColorSchemes.southwest, (jθ-1)/(Nθ-1)), line_dash="dash",name = "θ=$(@sprintf("%0.3g",θv))") for (jθ, θv) in enumerate(θvec), (jΔ, Δv) in enumerate(Δvec)][:]
			]
	end


	layout = Layout(template = template,
        title=title,
		# legend = attr(orientation="h"),
		# width = 1920*0.5, height=1080*0.25,
		# font_family="Linux Libertine", font_size = 16,
		xaxis=attr(title="<i>$sym"),
		yaxis=attr(title=ytitle))

	table = ""
	for (jθ, θv) in enumerate(θvec)
		if !rel_0
			table *= "$(@sprintf("%0.3g",θv))"
		end
		table *= "& $(@sprintf("%0.3g",round(CEmat[1, jθ, 1], digits=4)))\\% & $(@sprintf("%0.3g",maximum(CEmat[:, jθ, 1])))\\% & $(@sprintf("%0.3g",CE_FI[jθ, 1]))\\% \\\\ \n"
	end
	println(table)
	plot(scats, layout)
end

function plots_arrow(; γ=1, y1=0.75, βb=1, Nθ = 9, NΔ = 1, UCP = 1e-8, Nz = 51, lw = 2, slides = true, dark=!slides, template::Template=qtemplate(slides=slides, dark=dark), Nb = 101, Δ = 0.1, voxeu = false, IMFRP = false, saveres = false)

	rb, rf, γ, Δ, g = default_params(2)
	y1 = (1+g/100)^-5
	βb = (1+rb/100)^-5
	β  = (1+rf/100)^-5

	Δvec = range(Δ-(NΔ-1)*0.025, Δ+(NΔ-1)*0.025, length=NΔ)
	θvec = max.(1e-9, range(0, 4, length=Nθ))

	R_mat = zeros(Nθ, NΔ, Nz)
	sc = SCI(γ=γ, y1=y1, βb=βb, β = β, Nz = Nz, θ=θvec[1], Δ=Δvec[1])

	R0 = zeros(length(sc.gr[:z]))
	vtemp = -Inf
	bgrid = range(0, 2*0.1, length = Nb)
	for (jb, bv) in enumerate(bgrid)
		v, q1, def_prob, ER, dist_prob, densvec, def_vec, qθ_cont, qθ_def, q_RE, l_ratio = value_b(sc, bv, ones(length(sc.gr[:z])))
		if v > vtemp
			vtemp = v
			R0 = bv * (1 .- def_vec)
		end
	end

	# θvec = range(1e-9, 10, length=Nθ)
	for (jθ, θv) in enumerate(θvec), (jΔ, Δv) in enumerate(Δvec)
		sc.pars[:θ] = θv
		sc.pars[:Δ] = Δv
		
		vFI, RFI, _, mdp = arrow_securities(sc)
		
		R_mat[jθ, jΔ, :] = RFI
	end
	
	if saveres
		R_out = R_mat[:,1,:]

		R_out = [sc.gr[:z] R_out' 100*R0]

		CSV.write("R_IMFRP.csv", Tables.table(R_out, header = ["z", ["θ = $(θv)" for θv in θvec]..., "R0"]))
	end
	
	getcol(θv, θvec, dark) = getcol((θv-minimum(θvec))/(maximum(θvec)-minimum(θvec)), dark)
	getcol(x, dark) = get(ColorSchemes.davos, ifelse(dark, 0.15 + 0.85*x, 0.85*x))
	defcol(dark) = get(ColorSchemes.lajolla, 0.2*(dark) + 0.8*(1-dark))
	Rcol(dark) = ifelse(dark, "white", "black")

	colscale = [[vv, getcol(vv, dark)] for vv in range(0,1,length=Nθ)]

	cols = [(jθ-1)/(Nθ-1) for jθ in eachindex(θvec)][:]
	xs = [1 for jθ in eachindex(θvec)][:]
	ys = [0 for jθ in eachindex(θvec)][:]
	colnames = ["$(@sprintf("%0.3g", round(θv, digits=3)))" for θv in range(extrema(θvec)..., length=5)]

	ytitle = ""
	if voxeu

		colnames = ["Bajo", "Alto"]
		scats = [
			scatter(x=sc.gr[:z], y=R0, name = "no cont.", marker_color="black", line_dash="dash")
			[scatter(x = sc.gr[:z], y = R_mat[jθ, jΔ, :], showlegend=false, marker_color=colscale[jθ][2], name="<i>θ</i> = $θv"*ifelse(NΔ>1,", Δ=$Δv", "")) for (jθ, θv) in enumerate(θvec), (jΔ, Δv) in enumerate(Δvec)
						][:]
			scatter(mode = "markers", size = 0.25, marker_opacity=0,
				x = xs, y = ys, showlegend=false,
				marker = attr(color=cols, reversescale=false, colorscale=colscale, colorbar = attr(tickvals=range(0.1,0.9,length=length(colnames)), title="Robustez", ticktext=colnames, len = 0.9, y = 0.9, yanchor = "top")),
				)
		]
		xtitle = "<i>PBI"
		title = "Pago estipulado"
	elseif IMFRP
		colnames = ["Low", "High"]
		scats = [
			scatter(x=(sc.gr[:z] .- 1) * 100, y= 100 * R0, name = "noncont.", marker_color="black", line_dash="dash")
			[scatter(x = (sc.gr[:z] .- 1) * 100, y = 100 * R_mat[jθ, jΔ, :], showlegend=false, marker_color=colscale[jθ][2], name="<i>θ</i> = $θv"*ifelse(NΔ>1,", Δ=$Δv", "")) for (jθ, θv) in enumerate(θvec), (jΔ, Δv) in enumerate(Δvec)
						][:]
			scatter(mode = "markers", size = 0.25, marker_opacity=0,
				x = xs, y = ys, showlegend=false,
				marker = attr(color=cols, reversescale=false, colorscale=colscale, colorbar = attr(tickvals=range(0.1,0.9,length=length(colnames)), title="Robustness", ticktext=colnames, len = 0.9, y = 0.9, yanchor = "top")),
				)
		]
		xtitle = "<i>GDP</i> (in % deviation from baseline forecast, over 5 years)"
		ytitle = "Stipulated repayments (in % of GDP in baseline)"
		title = ""
	else
		scats = [
				scatter(x=sc.gr[:z], y=R0, name = "noncont.", marker_color=Rcol(dark), line_dash="dash")
				[scatter(x = sc.gr[:z], y = R_mat[jθ, jΔ, :], opacity = 0.8, line_width=lw, showlegend = false, marker_color=getcol(θv, θvec, dark), name="<i>θ</i> = $θv"*ifelse(NΔ>1,", Δ=$Δv", "")) for (jθ, θv) in enumerate(θvec), (jΔ, Δv) in enumerate(Δvec)
						][:]
				scatter(mode = "markers", size = 0.25, marker_opacity=0,
				x = xs, y = ys, showlegend=false,
				marker = attr(color=cols, reversescale=false, colorscale=colscale, colorbar = attr(tickvals=range(0.05,0.95,length=length(colnames)), title="&nbsp;&nbsp;<i>θ", ticktext=colnames, len = 0.9, y = 0.9, yanchor = "top")),
				)
			]
		xtitle = "<i>y</i><sub>2</sub>(<i>z</i>)"
		title = "<i>b R*</i>(<i>z;θ</i>)"
	end
	layout = Layout(template=template,
        legend = attr(x = 1.08, xref = "paper", xanchor="right",
			orientation="v", y = 1, yanchor="bottom", yref = "paper"),
        # legend = attr(orientation="v", xref="paper",xanchor="right",x=1.05,
            # y=0, yanchor="top"),
        title=title,
		xaxis=attr(title=xtitle),
		yaxis=attr(title=ytitle)
		)
	plot(scats, layout)
end

function plot_probs(; γ = 1, y1 = 0.75, βb = 1, β = βb, Δ = 0.1, Nθ = 5, NΔ = 1, UCP = 1e-8, Nz = 21, slides = true, dark=!slides, template::Template=qtemplate(slides=slides, dark=dark), bench = "", Nb = 101, g = 0, rb = 0, rf = 0, k = 0.5, α = 1, reopt = false, lw = 2, only_orig = false, only_OG_wdef = false, with_distort = false)

    if g != 0
        y1 = (1 + g / 100)^-5
    end
    if rb != 0
        βb = (1 + rb / 100)^-5
    end
    if rf != 0
        β = (1 + rf / 100)^-5
    end

    Δvec = range(Δ - (NΔ - 1) * 0.025, Δ + (NΔ - 1) * 0.025, length = NΔ)
    θvec = max.(1e-9, range(0, 4, length = Nθ))

	if bench == "1.5linear"
		α = 1.5
		bench = "linear"
	end

	if bench == "1.25linear"
		α = 1.25
		bench = "linear"
	end

    if bench == "optimal_RE" || bench == "noncontingent" || bench == "threshold" || bench == "linear" || bench == "linthres"
        sc = SCI(γ = γ, y1 = y1, βb = βb, β = β, Nz = Nz, θ = θvec[1], Δ = Δ)
    elseif bench == "robust"
        sc = SCI(γ = γ, y1 = y1, βb = βb, β = β, Nz = Nz, θ = θvec[end], Δ = Δ)
    else
        throw(error("Choose bench = 'optimal_RE', 'noncontingent', 'robust', 'threshold', 'linear', 'linthres"))
    end
    if bench == "noncontingent" || bench == "threshold" || bench == "linear" || bench == "linthres"
        bench == "noncontingent" ? kv = 0 : kv = k
        zthres = sc.gr[:z][findfirst(cumsum(sc.prob[:z]) .>= kv)]
        Rt = sc.gr[:z] .>= zthres
        bench == "linear" || bench == "linthres" ? Rt = 1 .+ α * (sc.gr[:z] .- mean(sc.gr[:z])) : nothing
        bench == "linthres" ? Rt = Rt .* (sc.gr[:z] .>= zthres) : nothing

        Rt = max.(0, Rt)

        R0 = zeros(length(sc.gr[:z]))
        vtemp = -Inf
        bgrid = range(0, 5 * Δ, step = 1e-3)

        b_opt, _, _, def_prob = optim_debt(sc, Rt)[1:4]
        R0 = 1 * Rt
    else
        v0, R0, _ = arrow_securities(sc)
        b_opt = (sc.prob[:z]' * R0)
        R0 = R0 / b_opt
    end

    distprob_vec = zeros(Nθ, NΔ)
    dens_vec = zeros(Nθ, NΔ, Nz)
    def_mat = zeros(Nθ, NΔ, Nz)
    likelihood = zeros(Nθ, NΔ, Nz)

    for (jθ, θv) in enumerate(θvec), (jΔ, Δv) in enumerate(Δvec)
        sc.pars[:θ] = θv
        sc.pars[:Δ] = Δv

        if reopt
            b_opt = optim_debt(sc, R0)[1]
        end

        _, _, def_prob, _, dist_prob, densvec, def_vec, qθ_cont, qθ_def, q_RE, l_ratio = value_b(sc, b_opt, R0)

        distprob_vec[jθ, jΔ] = dist_prob
        dens_vec[jθ, jΔ, :] = densvec
        def_mat[jθ, jΔ, :] = def_vec
        likelihood[jθ, jΔ, :] = l_ratio
    end

    getcol(θv, θvec, dark) = getcol((θv - minimum(θvec)) / (maximum(θvec) - minimum(θvec)), dark)
    getcol(x, dark) = get(ColorSchemes.davos, ifelse(dark, 0.15 + 0.85 * x, 0.8 * x))
    defcol(dark) = get(ColorSchemes.lajolla, 0.1 * (dark) + 0.8 * (1 - dark))
    Rcol(dark) = ifelse(dark, "white", "black")

    colscale = [[vv, getcol(vv, dark)] for vv in range(0, 1, length = Nθ)]

    cols = [(jθ - 1) / (Nθ - 1) for jθ in eachindex(θvec)][:]
    xs = [1 for jθ in eachindex(θvec)][:]
    ys = [0 for jθ in eachindex(θvec)][:]
    colnames = ["$(@sprintf("%0.3g", round(θv, digits=3)))" for θv in range(extrema(θvec)..., length = 5)]

    scats = [
        [scatter(x = sc.gr[:z], yaxis = "y1", y = dens_vec[jθ, 1, :] ./ maximum(dens_vec[1, 1, :]), line_width = 2lw, opacity = 0.75, marker_color = getcol(θv, θvec, dark), showlegend = false, name = "<i>θ</i> = $(@sprintf("%0.3g", θv))") for (jθ, θv) in enumerate(θvec) if jθ == 1]
        scatter(mode = "markers", size = 0.25, marker_opacity = 0, yaxis = "y1",
            x = xs, y = ys, showlegend = false,
            marker = attr(color = cols, reversescale = false, colorscale = colscale, colorbar = attr(tickvals = range(0.05, 0.95, length = length(colnames)), title = "&nbsp;&nbsp;<i>θ", ticktext = colnames, len = 0.9, y = 0.9, yanchor = "top")),
        )
        scatter(x = sc.gr[:z], yaxis = "y3", y = R0, marker_color = Rcol(dark), line_dash = "dash", name = "<i>R</i>(<i>z</i>)")
        scatter(yaxis = "y3")
    ]

    if !only_orig
        push!(scats, scatter(x = sc.gr[:z], y = def_mat[1, 1, :], opacity = 0.75, marker_color = defcol(dark), line_width = 2.5, hoverinfo = "skip", showlegend = (1 == 1), line_dash = "dot", yaxis = "y1", name = "def"))

        if !only_OG_wdef
            morescats = [scatter(x = sc.gr[:z], yaxis = "y3", y = likelihood[jθ, 1, :], line_width = lw, opacity = 0.75, marker_color = colscale[jθ][2], showlegend = false, name = "<i>θ</i> = $(@sprintf("%0.3g", θv))") for (jθ, θv) in enumerate(θvec)]

            if !with_distort
                push!(morescats, [scatter(x = sc.gr[:z], yaxis = "y1", y = dens_vec[jθ, 1, :] ./ maximum(dens_vec[1, 1, :]), line_width = lw, opacity = 0.75, marker_color = getcol(θv, θvec, dark), showlegend = false, name = "<i>θ</i> = $(@sprintf("%0.3g", θv))") for (jθ, θv) in enumerate(θvec)]...)

                scats = [morescats..., scats[2:end]...]
            else
                scats = [morescats..., scats...]
            end

        end
    end

    annots = [attr(text = "Note: Return of $bench debt", x = 0, y = -0.15, yanchor = "top", xref = "paper", yref = "paper", showarrow = false, font_family = "Linux Libertine", font_size = 14)]
    layout = Layout(template = template,
        title = "Distorted probabilities", #annotations = annots,
        # width = 1920*0.8, height=1080*0.8,
        legend = attr(x = 1.08, xref = "paper", xanchor="right",
			orientation="v", y = 0.85, yanchor="bottom", yref = "paper"),
        # legend = attr(xanchor="center", x = 0.5, xref = "paper"),
        # font_family="Linux Libertine", font_size = 16,
        xaxis = attr(anchor = "y3", zeroline = false, title = "<i>y</i><sub>2</sub>(<i>z</i>)"),
        yaxis1 = attr(domain = [0.35, 1], zeroline = false, title = ""),
        yaxis2 = attr(zeroline = false, overlaying = "y3", side = "right", range = [0, 2.5]),
        yaxis3 = attr(domain = [0, 0.25], zeroline = false, title = "", range = [0, 2.5]),
        # legend = attr(orientation="h", x=0.05),
        # yaxis2 = attr(titlefont_size=18, title="bps", overlaying = "y", side="right", zeroline=false),
    )


    plot(scats, layout)
end

function plot_spreads(; θmin = 1e-09, θmax = 4, Nθ = 26, γ = 5, g = 5, UCP = 1e-09, rb = 6, rf = 3, k = 0.5, αv = 0, Δ = 0.1, Nz = 501, slides = true, dark=!slides, template::Template=qtemplate(slides=slides, dark=dark), optimal_RE = false, robust = false)

	if robust
		optimal_RE = false
	end

	y1 = (1+g/100)^-5

	sprRE = Vector{Float64}(undef, Nθ)
	sprθC = Vector{Float64}(undef, Nθ)
	sprθD = Vector{Float64}(undef, Nθ)
	spr_def = Vector{Float64}(undef, Nθ)
	totspr = Vector{Float64}(undef, Nθ)
	defs = Vector{Float64}(undef, Nθ)
	CE   = Vector{Float64}(undef, Nθ)
	debt = Vector{Float64}(undef, Nθ)
	qbs   = Vector{Float64}(undef, Nθ)
	θvec = max.(θmin, range(0, θmax, length=Nθ))
	sc = SCI(γ = γ, y1 = y1, UCP = UCP, Nz = Nz, βb = (1+rb/100)^-5, β=(1+rf/100)^-5, Δ = Δ);

	zthres = sc.gr[:z][findfirst(cumsum(sc.prob[:z]).>= k)]

	R_purethres = sc.gr[:z] .>= zthres;
	R_slope = 1 .+ αv * (sc.gr[:z] .- mean(sc.gr[:z]))

	R = R_purethres .* R_slope

	title = "Debt with α = $αv, k = $k"

	if optimal_RE || robust
		optimal_RE ? θ_des = θmin : θ_des = 4
		sc1 = SCI(γ = γ, y1 = y1, UCP = UCP, Nz = 61, βb = (1+rb/100)^-5, β=(1+rf/100)^-5, Δ = Δ, θ = θ_des)
		v0, R1, _ = arrow_securities(sc1)
		itp_R = interpolate((sc1.gr[:z],), R1, Gridded(Linear()))
		meanR = sc1.prob[:z]'*R1
		R = itp_R(sc.gr[:z]) ./ meanR

		if optimal_RE
			title = "Debt designed for RE lenders"
		else
			title = "Debt designed for robust lenders"
		end
	end

	R_comp = ones(size(sc.gr[:z]))

	for (jθ, θv) in enumerate(θvec)
		sc.pars[:θ] = θv

		b_comp = optim_debt(sc, R_comp)[1];

		b_opt, v, q1, def_prob, ER, qθ_cont, qθ_def, q_RE, l_ratio = optim_debt(sc, R);

		sprRE[jθ], sprθC[jθ], sprθD[jθ], spr_def[jθ], totspr[jθ] = get_spreads(sc, R, q1, qθ_cont, qθ_def, q_RE) .* 10000
		qbs[jθ]   = q1 * b_opt
		# sprθC[jθ] = qθ_cont
		# sprθD[jθ]  = qθ_def
		defs[jθ] = def_prob
		debt[jθ] = b_opt
		CE[jθ]   = cons_equiv(sc, v, R_comp, b_comp)
	end

	sc = [
		scatter(x=θvec, y=round.(debt, digits = 10), name="debt", marker_color="#1D3557", xaxis="x2", yaxis="y2")
		scatter(x=θvec, y=100*defs, name = "def. prob (%, rhs)", marker_color="#E63946", xaxis="x2", yaxis="y3")
		scatter(x=θvec, y=qbs, name = "issuance value", marker_color="#2BA84A", xaxis="x3", yaxis="y4")
		scatter(x=θvec, y=round.(CE, digits=12), name = "Cons. equiv (%, rhs)", marker_color="#244F94", xaxis="x3", yaxis="y5")
		scatter(x=θvec, y=totspr, marker_color="black", showlegend=false, line_dash="dash", line_width=1, xaxis = "x1", yaxis = "y1", name="Total spread")
		bar(x=θvec, y=sprRE, marker_color="#322A26", xaxis = "x1", yaxis = "y1", name="R.E. spread")
		bar(x=θvec, y=sprθC, marker_color="#454B66", xaxis = "x1", yaxis = "y1", name="Ambig contingency")
		bar(x=θvec, y=sprθD, marker_color="#677DB7", xaxis = "x1", yaxis = "y1", name="Ambig default")
		]

	maxdef = max(4*UCP, maximum(100*defs)*1.05)

	annots = [
		attr(x = 2, y = -0.05, xref=xv,xanchor="center", yref="paper", yanchor="top", showarrow=false, text="<i>θ", font_size=18) for xv in ["x1", "x2"]
	]

	shapes = [hline(0, yref="y5", x0 = 0, x1 = 4, xref="x3", line_dash="dash")]

	layout = Layout(
        template = template,
        barmode="stack", legend = attr(orientation="h", xanchor="center", x = 0.5, xref="paper"),
		annotations = annots, shapes=shapes,
		width = 1920 * 0.6, height = 1080*0.6,
		xaxis1 = attr(zeroline=false, anchor="y1", domain = [0, 0.55]),
		xaxis2 = attr(zeroline=false, anchor="y2", domain = [0.65, 1]),
		xaxis3 = attr(zeroline=false, anchor="y4", domain = [0.65, 1]),
		yaxis1 = attr(anchor="x1",domain = [0, 1], zeroline=false, title="Spread (bps)"),
		yaxis2 = attr(anchor="x2",domain = [0, 0.45], zeroline=false),
		yaxis3 = attr(anchor="x2",domain = [0, 0.45], zeroline=false, side="right", overlaying="y2"),
		yaxis4 = attr(anchor="x3", domain = [0.55, 1], zeroline=false),
		yaxis5 = attr(anchor="x3", domain = [0.55, 1], zeroline=false, side="right", overlaying="y4"),
		title=title,
		)

	plot(sc, layout)
	# plot(scatter(x=θvec, y=qbs, name="q"))
	# plot(scatter(x=θvec, y=defs, name="def. prob"))
	# plot(scatter(x=θvec, y=debt, name="debt"))
	# ]
end

function plot_mixdebts(; Nϕ = 25, Nz = 501, slides = true, dark=!slides, template::Template=qtemplate(slides=slides, dark=dark))

	rb, rf, γ, Δ, g = default_params(2)
	if g != 0
		y1 = (1+g/100)^-5
	end
	if rb != 0
		βb = (1+rb/100)^-5
	end
	if rf != 0
		β=(1+rf/100)^-5
	end

	sc = SCI(γ = γ, y1 = y1, Nz = Nz, βb = βb, β=β, Δ = Δ, θ = 2);
	
	R_comp = ones(size(sc.gr[:z]))
	R_purethres = sc.gr[:z] .>= quantile(sc.gr[:z], 0.5);

	sprRE = Vector{Float64}(undef, Nϕ)
	sprθC = Vector{Float64}(undef, Nϕ)
	sprθD = Vector{Float64}(undef, Nϕ)
	spr_def = Vector{Float64}(undef, Nϕ)
	totspr = Vector{Float64}(undef, Nϕ)
	CE   = Vector{Float64}(undef, Nϕ)

	ϕvec = range(0,1,length=Nϕ)

	b_comp = optim_debt(sc, R_comp)[1];
	for (jϕ, ϕv) in enumerate(ϕvec)

		R = ϕv * R_purethres + (1-ϕv) * R_comp

		b_opt, v, q1, def_prob, ER, qθ_cont, qθ_def, q_RE, l_ratio = optim_debt(sc, R);

		sprRE[jϕ], sprθC[jϕ], sprθD[jϕ], spr_def[jϕ], totspr[jϕ] = get_spreads(sc, R, q1, qθ_cont, qθ_def, q_RE) .* 10000
		# qbs[jϕ]   = q1 * b_opt
		# sprθC[jϕ] = qθ_cont
		# sprθD[jϕ]  = qθ_def
		# defs[jϕ] = def_prob
		# debt[jϕ] = b_opt
		CE[jϕ]   = cons_equiv(sc, v, R_comp, b_comp)
	end

	layout = Layout(
        template = template,
		xaxis_title="Proportion of threshold bond"
		)

	plot(scatter(x=ϕvec, y=CE), layout)
end

function spreads_ϵ(; k = 0.5, Nz = 501, θmin = 1e-09, θmax = 4, Nθ = 25, slides = true, dark=!slides, template::Template=qtemplate(slides=slides, dark=dark))
	rb, rf, γ, Δ, g = default_params(2)
	if g != 0
		y1 = (1+g/100)^-5
	end
	if rb != 0
		βb = (1+rb/100)^-5
	end
	if rf != 0
		β=(1+rf/100)^-5
	end

	θvec = max.(θmin, range(0, θmax, length=Nθ))

	sprRE = Vector{Float64}(undef, Nθ)
	sprθC = Vector{Float64}(undef, Nθ)
	sprθD = Vector{Float64}(undef, Nθ)
	spr_def = Vector{Float64}(undef, Nθ)
	totspr = Vector{Float64}(undef, Nθ)

	for (jθ, θv) in enumerate(θvec)
		sc = SCI(γ = γ, y1 = y1, Nz = Nz, βb = βb, β=β, Δ = Δ, θ = θv);

		R_comp = ones(size(sc.gr[:z]))
		R_thres = sc.gr[:z] .>= quantile(sc.gr[:z], k);
		b_comp, q_comp = optim_debt(sc, R_comp)[[1,3]];

		q, qθ_cont, qθ_def, q_RE = price_otherdebt(sc, b_comp, R_comp, R_comp)

		sprRE[jθ], sprθC[jθ], sprθD[jθ], spr_def[jθ], totspr[jθ] = get_spreads(sc, R_comp, q, qθ_cont, qθ_def, q_RE) .* 10000
	end

	bars = [
		bar(x=θvec, y=sprRE, marker_color="#322A26", name="R.E. spread")
		bar(x=θvec, y=sprθC, marker_color="#677DB7", name="Ambig contingency")
		bar(x=θvec, y=sprθD, marker_color="#A1C6EA", name="Ambig default")
		]

	layout = Layout(
        template = template,
		barmode = "stack", font_size = 18,
		legend = attr(orientation="h")
	)
	plot(bars, layout)
end

function prep_spreads(; θvec, γ = 5, g = 5, UCP = 1e-09, rb = 6, rf = 3, k = 0.5, αv = 0, Δ = 0.1, Nz = 501, optimal_RE = false, robust = false)

	if robust
		optimal_RE = false
	end

	y1 = (1+g/100)^-5
	Nθ = length(θvec)

	sprRE = Vector{Float64}(undef, Nθ)
	sprθC = Vector{Float64}(undef, Nθ)
	sprθD = Vector{Float64}(undef, Nθ)
	spr_def = Vector{Float64}(undef, Nθ)
	totspr = Vector{Float64}(undef, Nθ)
	defs = Vector{Float64}(undef, Nθ)
	CE   = Vector{Float64}(undef, Nθ)
	debt = Vector{Float64}(undef, Nθ)
	qbs   = Vector{Float64}(undef, Nθ)
	sc = SCI(γ = γ, y1 = y1, UCP = UCP, Nz = Nz, βb = (1+rb/100)^-5, β=(1+rf/100)^-5, Δ = Δ);

	zthres = sc.gr[:z][findfirst(cumsum(sc.prob[:z]).>= k)]

	R_purethres = sc.gr[:z] .>= zthres;
	R_slope = 1 .+ αv * (sc.gr[:z] .- mean(sc.gr[:z]))

	R = R_purethres .* R_slope

	title = "Debt with α = $αv, k = $k"

	if optimal_RE || robust
		optimal_RE ? θ_des = first(θvec) : θ_des = 4
		sc1 = SCI(γ = γ, y1 = y1, UCP = UCP, Nz = 61, βb = (1+rb/100)^-5, β=(1+rf/100)^-5, Δ = Δ, θ = θ_des)
		v0, R1, _ = arrow_securities(sc1)
		itp_R = interpolate((sc1.gr[:z],), R1, Gridded(Linear()))
		meanR = sc1.prob[:z]'*R1
		R = itp_R(sc.gr[:z]) ./ meanR

		if optimal_RE
			title = "Debt designed for RE lenders"
		else
			title = "Debt designed for robust lenders"
		end
	end

	R_comp = ones(size(sc.gr[:z]))

	for (jθ, θv) in enumerate(θvec)
	# map(enumerate(θvec)) do (jθ, θv)
		sc.pars[:θ] = θv

		b_comp = optim_debt(sc, R_comp)[1];

		b_opt, v, q1, def_prob, ER, qθ_cont, qθ_def, q_RE, l_ratio = optim_debt(sc, R);

		sprRE[jθ], sprθC[jθ], sprθD[jθ], spr_def[jθ], totspr[jθ] = get_spreads(sc, R, q1, qθ_cont, qθ_def, q_RE) .* 10000
		qbs[jθ]   = q1 * b_opt
		# sprθC[jθ] = qθ_cont
		# sprθD[jθ]  = qθ_def
		defs[jθ] = def_prob
		debt[jθ] = b_opt
		CE[jθ]   = cons_equiv(sc, v, R_comp, b_comp)
	end

	return sprRE, sprθC, sprθD, qbs, CE
end

function plot_optimal(; θmin = 1e-09, θmax = 4, Nθ = 26, γ = 5, g = 5, UCP = 1e-09, rb = 6, rf = 3, Δ = 0.1, Nz = 501, slides = true, dark=!slides, template::Template=qtemplate(slides=slides, dark=dark))
	rb, rf, γ, Δ, g = default_params(2)
	namevec = ["optimal_RE", "robust"]
	Nv = length(namevec)

	sprRE = Vector{Array{Float64}}(undef, length(namevec))
	sprθC = Vector{Array{Float64}}(undef, length(namevec))
	sprθD = Vector{Array{Float64}}(undef, length(namevec))
	qbs = Vector{Array{Float64}}(undef, length(namevec))
	CEs = Vector{Array{Float64}}(undef, length(namevec))

	θvec = max.(θmin, range(0, θmax, length=Nθ))

	sprRE[1], sprθC[1], sprθD[1], qbs[1], CEs[1] = prep_spreads(θvec = θvec, γ = γ, g = g, rb = rb, rf = rf, Δ = Δ, Nz = Nz, optimal_RE = true)
	sprRE[2], sprθC[2], sprθD[2], qbs[2], CEs[2] = prep_spreads(θvec = θvec, γ = γ, g = g, rb = rb, rf = rf, Δ = Δ, Nz = Nz, robust = true)
	

	getcol(x, dark) = get(ColorSchemes.oslo, ifelse(dark, 0.3 + 0.7*x, 0.75*x))
	defcol(x) = get(ColorSchemes.cork, x)

	sc = [
		[scatter(x=θvec, y=round.(CEs[jj], digits=12), showlegend = (jj==1), name = "Cons. equiv (%, rhs)", marker_color=defcol(0.2), line_width=3, xaxis="x$(Nv+jj)", yaxis="y$(2Nv+jj)") for jj in eachindex(namevec)]
		[scatter(x=θvec, y=qbs[jj], showlegend = (jj==1), name = "issuance value", marker_color=defcol(0.8), line_width=3, xaxis="x$(Nv+jj)", yaxis="y$(Nv+jj)") for jj in eachindex(namevec)]
		[bar(x=θvec, y=sprRE[jj], marker_color=getcol(0, dark), xaxis = "x$(jj)", yaxis = "y$(jj)", showlegend = (jj==1), name="R.E. spread") for jj in eachindex(namevec)]
		[bar(x=θvec, y=sprθC[jj], marker_color=getcol(0.5, dark), xaxis = "x$(jj)", yaxis = "y$(jj)", showlegend = (jj==1), name="Ambig contingency") for jj in eachindex(namevec)]
		[bar(x=θvec, y=sprθD[jj], marker_color=getcol(1, dark), xaxis = "x$(jj)", yaxis = "y$(jj)", showlegend = (jj==1), name="Ambig default") for jj in eachindex(namevec)]
	]

	spr_min = 0
	spr_max = 1.05 * maximum([maximum(sprRE[jj] + sprθC[jj] + sprθD[jj]) for jj in eachindex(sprRE)])

	qbs_min = minimum([minimum(qbs[jj]) for jj in eachindex(qbs)])
	qbs_max = 1.05 * maximum([maximum(qbs[jj]) for jj in eachindex(qbs)])

	wmin = 1.05 * minimum([minimum(CEs[jj]) for jj in eachindex(CEs)])
	wmax = 1.05 * maximum([maximum(CEs[jj]) for jj in eachindex(CEs)])

	annots = [
		attr(text = "Debt designed for RE lenders", x = mean(θvec), font_size = 22, 
			xref = "x1", xanchor="center", y = 1.05, yref = "paper", showarrow=false)
		attr(text = "Debt designed for robust lenders", x = mean(θvec), font_size = 22, 
			xref = "x2", xanchor="center", y = 1.05, yref = "paper", showarrow=false)
		[attr(x = 2, y = -0.05, xref=xv,xanchor="center", yref="paper", yanchor="top", showarrow=false, text="<i>θ", font_size=18) for xv in ["x3", "x4"]]
	]

	layout = Layout(template = template,
        annotations = annots,
		barmode = "stack", font_size = 18,
		legend = attr(orientation="h", xanchor="center", x = 0.5, xref="paper"),
		width = 1920 * 0.6, height = 1080*0.55,
		xaxis1 = attr(zeroline=false, anchor="y1", domain = [0.18, 0.47]),
		xaxis2 = attr(zeroline=false, anchor="y2", domain = [0.53, 0.82]),
		yaxis1 = attr(range=[spr_min, spr_max], zeroline=false, anchor="x1", title="Spread (bps)", titlefont_size=18, domain = [0.55, 1]),
		yaxis2 = attr(range=[spr_min, spr_max], zeroline=false, anchor="x2", domain = [0.55, 1]),
		xaxis3 = attr(zeroline=false, anchor="y3", domain = [0.18, 0.47]),
		xaxis4 = attr(zeroline=false, anchor="y4", domain = [0.53, 0.82]),
		yaxis3 = attr(range=[qbs_min, qbs_max], zeroline=false, anchor="x4", domain = [0, 0.45]),
		yaxis4 = attr(range=[qbs_min, qbs_max], zeroline=false, anchor="x5", domain = [0, 0.45]),
		yaxis5 = attr(range=[wmin, wmax], zeroline=true, zerolinecolor="#969696", zerolinewidth=2, anchor="x3", overlaying="y3", side="right"),
		yaxis6 = attr(range=[wmin, wmax], zeroline=true, zerolinecolor="#969696", zerolinewidth=2, anchor="x4", overlaying="y4", side="right"),
		)

	plot(sc, layout)
end
function plot_parametric(; θmin = 1e-09, θmax = 4, Nθ = 26, γ = 5, g = 5, UCP = 1e-09, rb = 6, rf = 3, Δ = 0.1, Nz = 501, slides = true, dark=!slides, template::Template=qtemplate(slides=slides, dark=dark))
	rb, rf, γ, Δ, g = default_params(2)

	namevec = ["noncontingent", "linear", "threshold"]
	Nv = length(namevec)
	αvec = [0, 1, 0.]
	kvec = [0, 0, 0.5]


	sprRE = Vector{Array{Float64}}(undef, length(namevec))
	sprθC = Vector{Array{Float64}}(undef, length(namevec))
	sprθD = Vector{Array{Float64}}(undef, length(namevec))
	qbs = Vector{Array{Float64}}(undef, length(namevec))
	CEs = Vector{Array{Float64}}(undef, length(namevec))

	θvec = max.(θmin, range(0, θmax, length=Nθ))

	for jj in eachindex(namevec)
		αv = αvec[jj]
		kv = kvec[jj]
		sprRE[jj], sprθC[jj], sprθD[jj], qbs[jj], CEs[jj] = prep_spreads(θvec = θvec, γ = γ, g = g, rb = rb, rf = rf, k = kv, αv = αv, Δ = Δ, Nz = Nz)
	end

	getcol(x, dark) = get(ColorSchemes.oslo, ifelse(dark, 0.3 + 0.7*x, 0.75*x))
	defcol(x) = get(ColorSchemes.cork, x)

	sc = [
		[scatter(x=θvec, y=round.(CEs[jj], digits=12), showlegend = (jj==1), name = "Cons. equiv (%, rhs)", marker_color=defcol(0.2), line_width=3, xaxis="x$(Nv+jj)", yaxis="y$(2Nv+jj)") for jj in eachindex(namevec)]
		[scatter(x=θvec, y=qbs[jj], showlegend = (jj==1), name = "issuance value", marker_color=defcol(0.8), line_width=3, xaxis="x$(Nv+jj)", yaxis="y$(Nv+jj)") for jj in eachindex(namevec)]
		[bar(x=θvec, y=sprRE[jj], marker_color=getcol(0, dark), xaxis = "x$(jj)", yaxis = "y$(jj)", showlegend = (jj==1), name="R.E. spread") for jj in eachindex(namevec)]
		[bar(x=θvec, y=sprθC[jj], marker_color=getcol(0.5, dark), xaxis = "x$(jj)", yaxis = "y$(jj)", showlegend = (jj==1), name="Ambig contingency") for jj in eachindex(namevec)]
		[bar(x=θvec, y=sprθD[jj], marker_color=getcol(1, dark), xaxis = "x$(jj)", yaxis = "y$(jj)", showlegend = (jj==1), name="Ambig default") for jj in eachindex(namevec)]
	]

	spr_min = 0
	spr_max = 1.05 * maximum([maximum(sprRE[jj] + sprθC[jj] + sprθD[jj]) for jj in eachindex(sprRE)])

	qbs_min = minimum([minimum(qbs[jj]) for jj in eachindex(qbs)])
	qbs_max = 1.05 * maximum([maximum(qbs[jj]) for jj in eachindex(qbs)])

	wmin = 1.05 * minimum([minimum(CEs[jj]) for jj in eachindex(CEs)])
	wmax = 1.05 * maximum([maximum(CEs[jj]) for jj in eachindex(CEs)])

	annots = [
		attr(text = "Noncontingent", x = mean(θvec), font_size = 22,
		# font_family = "Linux Libertine",
		xref = "x1", xanchor="center", y = 1.025, yref = "paper", showarrow=false)
		attr(text = "Linear", x = mean(θvec), font_size = 22,
		# font_family = "Linux Libertine",
		xref = "x2", xanchor="center", y = 1.025, yref = "paper", showarrow=false)
		attr(text = "Threshold", x = mean(θvec), font_size = 22,
		# font_family = "Linux Libertine",
		xref = "x3", xanchor="center", y = 1.025, yref = "paper", showarrow=false)
		[attr(x = 2, y = -0.05, xref=xv,xanchor="center", yref="paper", yanchor="top", showarrow=false, text="<i>θ", font_size=18) for xv in ["x4", "x5", "x6"]]
	]


	layout = Layout(template = template,
        annotations = annots,
		barmode = "stack", font_size = 18,
		legend = attr(orientation="h", xanchor="center", x = 0.5, xref="paper"),
		width = 1920 * 0.6, height = 1080*0.6,
		xaxis1 = attr(zeroline=false, anchor="y1", domain = [0, 0.33-0.04]),
		xaxis2 = attr(zeroline=false, anchor="y2", domain = [0.33+0.02, 0.67-0.02]),
		xaxis3 = attr(zeroline=false, anchor="y3", domain = [0.67+0.04, 1]),
		yaxis1 = attr(range=[spr_min, spr_max], zeroline=false, anchor="x1", title="Spread (bps)", titlefont_size=18, domain = [0.55, 1]),
		yaxis2 = attr(range=[spr_min, spr_max], zeroline=false, anchor="x2", domain = [0.55, 1]),
		yaxis3 = attr(range=[spr_min, spr_max], zeroline=false, anchor="x3", domain = [0.55, 1]),
		xaxis4 = attr(zeroline=false, anchor="y4", domain = [0, 0.33-0.04]),
		xaxis5 = attr(zeroline=false, anchor="y5", domain = [0.33+0.02, 0.67-0.02]),
		xaxis6 = attr(zeroline=false, anchor="y6", domain = [0.67+0.04, 1]),
		yaxis4 = attr(range=[qbs_min, qbs_max], zeroline=false, anchor="x4", domain = [0, 0.45]),
		yaxis5 = attr(range=[qbs_min, qbs_max], zeroline=false, anchor="x5", domain = [0, 0.45]),
		yaxis6 = attr(range=[qbs_min, qbs_max], zeroline=false, anchor="x6", domain = [0, 0.45]),
		yaxis7 = attr(range=[wmin, wmax], zeroline=true, zerolinecolor="#969696", zerolinewidth=2, anchor="x4", overlaying="y4", side="right"),
		yaxis8 = attr(range=[wmin, wmax], zeroline=true, zerolinecolor="#969696", zerolinewidth=2, anchor="x5", overlaying="y5", side="right"),
		yaxis9 = attr(range=[wmin, wmax], zeroline=true, zerolinecolor="#969696", zerolinewidth=2, anchor="x6", overlaying="y6", side="right"),
		)

	plot(sc, layout)
end

function default_params(bench = 1)
	if bench == 1
		rb = 10
		rf = 3
		γ = 6
		Δ = 0.2
		g = 10
	elseif bench == 2
		rb = 6
		rf = 3
		γ = 2
		Δ = 0.2
		g = 8
	end

	return rb, rf, γ, Δ, g
end

function save_all_probs(;slides=true, dark = false, Nθ = 9)
	jsli  = ifelse(slides, "_slides", "_paper")

	dark ? jsli *= "_dark" : nothing
	lw = ifelse(slides, 2, 3)

	namevec = ["noncontingent", "threshold", "linear", 
	# "1.25linear", "1.5linear",
	"optimal_RE", "robust", "linthres"]

	rb, rf, γ, Δ, g = default_params(2)
	for (jn, nv) in enumerate(namevec)
		if nv == "optimal_RE" || nv == "robust"
			Nz = 41
		else
			Nz = 5001
		end

		reopts = ifelse(slides, [false, true], [false])
		for reopt in reopts
			if jn == 1 && !reopt && slides
				p1 = plot_probs(rb=rb, rf = rf, γ = γ, Δ=Δ, g=g, bench=nv, Nθ = Nθ, Nz = Nz, reopt=reopt, dark=dark, slides=slides, lw=lw, only_orig = true)

				savefig(p1, "output/distorted_only_orig$(jsli).pdf", width = qwidth(slides), height=qheight(slides))

				p1 = plot_probs(rb=rb, rf = rf, γ = γ, Δ=Δ, g=g, bench=nv, Nθ = Nθ, Nz = Nz, reopt=reopt, dark=dark, slides=slides, lw=lw, only_OG_wdef = true)

				savefig(p1, "output/distorted_only_OG_wdef$(jsli).pdf", width = qwidth(slides), height=qheight(slides))

				p1 = plot_probs(rb=rb, rf = rf, γ = γ, Δ=Δ, g=g, bench=nv, Nθ = Nθ, Nz = Nz, reopt=reopt, slides=slides, dark=dark, lw=lw, with_distort = true)

				savefig(p1, "output/distorted_only_with_distort$(jsli).pdf", width = qwidth(slides), height=qheight(slides))
			end

			p1 = plot_probs(rb=rb, rf = rf, γ = γ, Δ=Δ, g=g, bench=nv, Nθ = Nθ, Nz = Nz, reopt=reopt, slides=slides, dark=dark, lw=lw)

			reopt ? suf = "_reopt" : suf = ""
			savefig(p1, "output/distorted_$nv"*suf*"$(jsli).pdf", width = qwidth(slides), height=qheight(slides))
		end
	end
	p1 = plots_arrow(Nz = 101, Nθ = Nθ, slides=slides, dark=dark, lw=lw+0.5)
	savefig(p1, "output/arrow$(jsli).pdf", width = qwidth(slides), height=qheight(slides))
end

function save_all_spreads(;slides=true, dark = false, Nz = 5001, Nz_opt = 501)
	jsli = ifelse(slides, "_slides", "_paper")
	dark ? jsli *= "_dark" : nothing

	p1 = plot_parametric(Nz = Nz, slides=slides, dark = dark)
	savefig(p1, "output/spreads_parametric$(jsli).pdf", width = ceil(Int,1.25*qwidth(slides)), height=ceil(Int,1.15*qheight(true)))

	p1 = plot_optimal(Nz = Nz_opt, slides=slides, dark = dark)
	savefig(p1, "output/spreads_optimal$(jsli).pdf", width = ceil(Int,1.25*qwidth(slides)), height=ceil(Int,1.15*qheight(true)))

	nothing
end

function save_all()
	for slides in [false, true]
		darks = [false]
		if slides 
			darks = [true, false]
		end
		for dark in darks
			Nθ = ifelse(slides, 9, 5)
			save_all_probs(slides=slides, dark=dark, Nθ = Nθ)
			save_all_spreads(slides=slides, dark=dark)
		end
	end
end