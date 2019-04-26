#using Plots
include("./BdG.jl")
import .Bdgeq

nc = 1000 #Number of Chebyshev polynomials that we consider
Nx = 24 #System size
Ny = 24 #System size
aa = 10.0 #Chebyshev renormalize parameter
bb = 0.0
ωc = 10.0 #Cutoff energy
U = -2.2 #Interaction
initialΔ = 0.1 #Initial guess for the superconducting gap
μ = -1.5 #Chemical potential
mφ = 1

println("Chebyshev method")
full = false #Full diagonalization or not
vortex = true
itemax = 20
@time mat_Δ = Bdgeq.iteration(nc,Nx,Ny,aa,bb,ωc,U,initialΔ,μ,full,vortex,itemax,mφ)
println("Full diagonalization")
full = true #Full diagonalization or not
vortex = true
itemax = 20
@time mat_Δ = Bdgeq.iteration(nc,Nx,Ny,aa,bb,ωc,U,initialΔ,μ,full,vortex,itemax,mφ)
println("Done.")


