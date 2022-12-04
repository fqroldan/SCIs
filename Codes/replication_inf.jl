include("main_sci_inf.jl")

# Point input to folder containing the JLD2 files
input = "Output/"
# As before ensure folder "folder" exists
folder = "Output/"

dd, dd_RE = load(input * "/dd.jld2", "dd", "dd_RE");

# Table 3: Calibration
tab3 = param_table(dd)
write(folder * "/tab3.txt", tab3)

# Table 4: Data versus model moments
tab4 = calib_table(dd, dd_RE)
write(folder * "/tab4.txt", tab4)
