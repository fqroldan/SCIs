include("sci_2p_v17.jl")

# Enter folder to save plots (make sure the folder exists)
folder = "Output/"

Nθ = 9
Nz = 5001

rb, rf, γ, Δ, g = default_params(2)

# Table 2: Parametrization
tab2 = write_tab2(rb, rf, γ, Δ, g)

write(folder * "/tab2.txt", tab2)

# Figure 2: distorted probabilities with noncontingent
fig2 = plot_probs(rb=rb, rf = rf, γ = γ, Δ=Δ, g=g, bench="noncontingent", Nθ = Nθ, Nz = Nz, reopt=false, slides=false, dark=false, lw=3)

savefig(fig2, folder * "/fig2.pdf", width = 864, height = 378)

# Figure 3: distorted probabilities with threshold
fig3 = plot_probs(rb=rb, rf = rf, γ = γ, Δ=Δ, g=g, bench="threshold", Nθ = Nθ, Nz = Nz, reopt=false, slides=false, dark=false, lw=3)

savefig(fig3, folder * "/fig3.pdf", width = 864, height = 378)

# Figure 4: Optimal design for each lender
fig4 = plots_arrow(Nz = 101, Nθ = 5, slides=false, dark=false, lw=3.5)

savefig(fig4, folder * "/fig4.pdf", width = 864, height = 378)


# Figure 5: Spreads and welfare
fig5 = plot_parametric(Nz = Nz, slides=false, dark=false)

savefig(fig5, folder * "/fig5.pdf", width = 1080, height = 621)
