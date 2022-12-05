include("main_sci_inf.jl")

# Point input to folder containing the JLD2 files
input = "Output/"
# As before ensure folder "folder" exists
folder = "Output/"

dd = load(input * "/dd.jld2", "dd");
dd_RE = load(input * "/dd_RE.jld2", "dd");

# Table 3: Calibration
tab3 = param_table(dd)
write(folder * "/tab3.txt", tab3)

# Table 4: Data versus model moments
tab4 = calib_table(dd, dd_RE)
write(folder * "/tab4.txt", tab4)

# Table 5: Different bond structures
dd_vec = [load(input * "/dd_RE.jld2", key) for key in ("dd", "dd_threshold", "dd_linear")];
tab5a = comp_table(dd_vec, rownames=true)

dd_vec = [load(input * "/dd.jld2", key) for key in ("dd", "dd_threshold", "dd_linear")];
tab5b = comp_table(dd_vec)

write(folder * "/tab5.txt", tab5a * "\n" * tab5b)

# Table 6: Spreads of marginal issuances
dd_opt = load(input * "/dd.jld2", "dd_opt")
tab6a = MTI_all(dd; α=dd_opt.pars[:α], τ=dd_opt.pars[:τ], showtop=true)

dd_opt = load(input * "/dd_RE.jld2", "dd_opt")
tab6b = MTI_all(dd_RE; α=dd_opt.pars[:α], τ=dd_opt.pars[:τ], showtop=false)

write(folder * "/tab6.txt", tab6a * tab6b)

# Table 7: Comparisons with optimal bonds
dd_vec = [load(input * "/dd_RE.jld2", key) for key in ("dd", "dd_opt")];
tab7a = comp_table(dd_vec, rownames=true)

dd_vec = [load(input * "/dd.jld2", key) for key in ("dd", "dd_opt")];
tab7b = comp_table(dd_vec, rownames=false)

write(folder * "/tab7.txt", tab7a * "\n" * tab7b)

# Table 8: Decomposition of welfare gains
dd_opt = load(input * "/dd.jld2", "dd_opt")
tab8a = welfare_decomp_opt(dd, dd_opt; showtop=true)

dd_opt_RE = load(input * "/dd_RE.jld2", "dd_opt")
tab8b = welfare_decomp_opt(dd_RE, dd_opt_RE; showtop=false)

write(folder * "/tab8.txt", tab8a * tab8b)

# Table 9: Different bond structures with recovery
dd_vec = [load(input * "/dd_recovery_RE.jld2", key) for key in ("dd", "dd_threshold", "dd_linear")];
tab9a = comp_table(dd_vec, rownames=true)

dd_vec = [load(input * "/dd_recovery.jld2", key) for key in ("dd", "dd_threshold", "dd_linear")];
tab9b = comp_table(dd_vec)

write(folder * "/tab9.txt", tab9a * "\n" * tab9b)

# Table 10: optimal bond with recovery
dd_vec = [load(input * "/dd_recovery_RE.jld2", key) for key in ("dd", "dd_opt")];
tab10a = comp_table(dd_vec, rownames=true)

dd_vec = [load(input * "/dd_recovery.jld2", key) for key in ("dd", "dd_opt")];
tab10b = comp_table(dd_vec, rownames=false)

write(folder * "/tab10.txt", tab10a * "\n" * tab10b)

# Table 11: calibration with long-run moments
tab11 = calib_table(dd, dd_RE, longrun=true)

write(folder * "/tab11.txt", tab11)

# Table 12: Calibration with recovery
dd_h04 = load(input * "/dd_recovery.jld2", "dd")
tab12 = param_table(dd_h04)
write(folder * "/tab12.txt", tab12)
