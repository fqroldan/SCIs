# Uncertainty Premia, Sovereign Default Risk, and State-Contingent Debt

Francisco Roch (IMF) and Francisco Roldán (IMF)

December 15, 2022

Version submitted to JPE: Macroeconomics

## Replication package

Codes are tested on Julia 1.7 and 1.8. TOML files are provided to ensure you have the same environment as the version which we ran for the paper. (Otherwise, you will need to install packages QuantEcon, Optim, Interpolations, LinearAlgebra, PlotlyJS, ColorSchemes, CSV, DataFrames, NLopt, ForwardDiff, Distributions, Printf, Random, StatsBase, Dates, JLD2.)

To run, point Julia to the SCI folder and ensure that "Codes", "Input", and "Output" are subfolders.

## Codes

Subfolder Codes contains all Julia programs needed to replicate the paper, as described below. Subfolder Input contains Julia JLD2 files containing solved versions of the relevant models. Subfolder Output contains all figures and tables as produced by our codes, except for Figure 1 which is taken from Igan et al (2021) and consists of fig1a.pdf, fig1b.pdf, and fig1c.pdf.

### Two-period model (Section 3)
Running 
```julia
include("Codes/replication_2p.jl")
```
will produce tab2.txt and fig2.pdf, fig3.pdf, fig4.pdf, and fig5.pdf corresponding to Table 2 and Figures 2 through 5 in the paper, for the two-period model.

### Infinite-horizon model (Sections 4 and 5)
Running 
```julia
include("Codes/replication_inf.jl")
```
will produce tab3.txt, tab4.txt, tab5.txt, tab6.txt, tab7.txt, tab8.txt, tab9.txt, tab10.txt, tab11.txt, tab12.txt, corresponding to Tables 3 through 9 in Section 5 and Tables 11 and 12 in the Appendix.