using Distributed
include("./Rscgsolver.jl")


#include("/Users/nagai/git/SS.jl/SS/src/SS.jl")

@everywhere module BdG
    using SparseArrays
    using Arpack
    using LinearAlgebra
    using Distributed
    using FFTW
    using Random

    #using ..SS




    using ..Rscgsolver

    Random.seed!(1234)
    #export set_Lattice
    #2D is only supported
    const dim=2
    const c_SI= 299792458 #speed of light in SI units
    const α = 2.18769126277e6/c_SI

    global const c = 1/α #speed of light in atomic units
    global const m_e = 1.0 #mass of electrons
    global const a_0 = 1.0 #Bohr length
    global const elec = 1.0 #elementary charge
    global const Eh = 1.0 #Hartree energy
    global const hbar = 1.0 #reduced Planck constant
    global const φ0 = π #magnetic quanta

    #println(c)
    mutable struct BdGsystem
        meshes::Array{Int64,1} #Nums of meshes
        Vol::Int64 #total numbers of sites
        L::Array{Float64,1} #System size in real space
        dr::Array{Float64,1} #dx and dy
        mass::Float64 #mass of electrons
        #order::Int8 #Accuracy of the differencial equation
        indexofcoffs_2nd::Array{Int64,2} #coefficients of 2nd derivative. dx and dy in the 1st dimension.
        #The second dimension inducates the positions like -1,0,1.
        valueofcoffs_2nd::Array{Float64,2} #coefficients of 2nd derivative
        indexofcoffs_1st::Array{Int64,2} #coefficients of 1st derivative. dx and dy in the 1st dimension.
        #The second dimension inducates the positions like -1,0,1.
        valueofcoffs_1st::Array{Float64,2} #coefficients of 1st derivative
        Δ::Array{Number,1} #Only for s-wave SC now.
        μ::Float64 #Chemical potential
        real::Bool #The Hamiltonian is real or not.
        mφ::Int8 #Vorticity in the unit cell
        calc_current::Bool #Calculate current or not
        current::Array{Float64,3} #current
        vec_A::Array{Float64,3} #Vector potential
        density::Array{Float64,2} #electron density
        Φ::Array{Float64,2} #Scalar potential
        use_Φ::Bool #use scalar potential or not
        d::Float64 #width
        potential::Array{Float64,2} #impurity potentials
        nimp::Float64
    end

    function set_Lattice(meshes,L;mass=1.0,order=2,real=true,Δ0 = 1.0,μ=0.5,mφ=0,calc_current=false,use_Φ=true,d=10.0,nimp=0,V0 = 1.0,ξv=0.1,nseed=1234)
        dr = L./(meshes)
#        dr = L./(meshes .- 1)
        Vol = prod(meshes)
        indexofcoffs_2nd,valueofcoffs_2nd = call_coefficients_2nd(dr,order)
        indexofcoffs_1st,valueofcoffs_1st = call_coefficients_1st(dr,order)
        if real
            Δ = Δ0*ones(Float64,Vol)
        else
            Δ = Δ0*ones(ComplexF64,Vol)
        end
        pF = 0.5
        pF = sqrt(2mass*μ)
        maxE =(1/2mass)*(4dim/minimum(dr)^2)
        println("System size: $(L[1]) [bohr] x $(L[2]) [bohr]")
        println("Mesh size: $(meshes[1]) x $(meshes[2]) ")
        println("μ = ",μ," [Hartree]")
        println("Maximum energy scale = ",maxE," [Hartree]")
#        println("Estimated minimum energy scale = ",maxE/meshes[1]," [Hartree]")
        println("Smallest length scale = ",minimum(dr)," [bohr]")

        println("Estimated smallest physical length scale 1/kF = ",1/pF," [bohr]")
        if mφ != 0
            real = false
        end
        if real
            println("The Hamiltonian is a real matrix")
        else
            println("The Hamiltonian is a complex matrix")
            if mφ !=0
                println("There are $mφ vortices in the unit cell")
            end
        end

        if calc_current
            println("We calculate the current distributions")
            current = zeros(Float64,meshes[1],meshes[2],2)
        else
            current = zeros(Float64,1,1,2)
        end

        Random.seed!(nseed)

        if maxE < μ
            println("μ is too large! μ should be smaller than $maxE in this system")
#            return
        end
        vec_A = zeros(Float64,meshes[1],meshes[2],2)
        density = zeros(Float64,meshes[1],meshes[2])
        Φ= zeros(Float64,meshes[1],meshes[2])
        #vec_A = init_A(mφ,L,meshes)

        Nx = meshes[1]
        Ny = meshes[2]
        if mφ == 2
            for ix=1:Nx
                for iy=1:Ny
                    ii = (iy-1)*Nx+ix
                    rx,ry = calc_r(ix,iy,Nx,Ny)
                    φ = atan(ry,rx)
                    φ +=  (atan(ry+(Ny)/2,rx+(Nx)/2)+atan(ry-(Ny)/2,rx-(Nx)/2)
                    +atan(ry+(Ny)/2,rx-(Nx)/2)+atan(ry-(Ny)/2,rx+(Nx)/2))

                    Δ[ii]= Δ[ii]*exp(-im*φ)
                    #println(Δ[ii])
                end
            end
        end

        if mφ == 8

            xx= rand(mφ)*Nx
            yy = rand(mφ)*Ny



            for ix=1:Nx
                for iy=1:Ny
                    ii = (iy-1)*Nx+ix
                    rx,ry = calc_r(ix,iy,Nx,Ny)
                    #φ = rand()*2π
                    φ = atan(ry,rx)
                    φ +=  (atan(ry+(Ny)/2,rx+(Nx)/2)+atan(ry-(Ny)/2,rx-(Nx)/2)
                    +atan(ry+(Ny)/2,rx-(Nx)/2)+atan(ry-(Ny)/2,rx+(Nx)/2))
                    φ +=  (atan(ry,rx+0.1+(Nx)/2)+#+atan(ry,rx-(Nx)/2)
                    atan(ry-0.1+(Ny)/2,rx))#+atan(ry-(Ny)/2,rx))
                    φ +=  (atan(ry+(Ny)/4,rx+(Nx)/4)+atan(ry-(Ny)/4,rx-(Nx)/4)
                    +atan(ry+(Ny)/4,rx-(Nx)/4)+atan(ry-(Ny)/4,rx+(Nx)/4))

                    #φ = 0#atan(ry,rx)
                    #for m=1:8
                    #    φ +=atan(ry-yy[m],rx-xx[m])
                    #end
                    #φ +=  (atan(ry+(Ny)/2,rx+(Nx)/2)+atan(ry-(Ny)/2,rx-(Nx)/2)
                    #+atan(ry+(Ny)/2,rx-(Nx)/2)+atan(ry-(Ny)/2,rx+(Nx)/2))

                    Δ[ii]= Δ[ii]*exp(-im*φ)
                    #println(Δ[ii])
                end
            end
        end



        potential = make_potentials(Nx,Ny,L,dr,V0,nimp,ξv)

        lattice = BdGsystem(meshes,Vol,L,dr,mass,
            indexofcoffs_2nd,valueofcoffs_2nd,indexofcoffs_1st,valueofcoffs_1st,Δ,μ,
            real,mφ,calc_current,current,vec_A,density,Φ,use_Φ,d,potential,nimp)
        flush(stdout)
        return lattice
    end

    function make_potentials(Nx,Ny,L,dr,V0,nimp,ξv)
        potential = zeros(Float64,Nx,Ny)
        if nimp == 0
            return potential
        end
        println("Impurities. nimp = $nimp, V0 = $V0")

        #=
        numV = round(Int64,L[1]*L[2]*nimp)
        println("$numV impurities are added")
        x0 = rand(Float64,numV).*L[1] .-L[1]/2
        y0 = rand(Float64,numV).*L[2] .-L[2]/2


        function calc_V(x,y,x0,y0,numV,ξv,V0)
            vsum = 0.0
            for i=1:numV
                vsum += V0*exp(-((x-x0[i])^2+(y-y0[i])^2)/ξv^2)
            end
            return vsum
        end
        =#

        for ix=1:Nx
            for iy=1:Ny
                #i = (iy-1)*Nx + ix
                #x = (ix-1)*dr[1]-L[1]/2
                #y = (iy-1)*dr[2]-L[2]/2
                if rand() < nimp
                    potential[ix,iy] = V0
                end
                #potential[i] = calc_V(x,y,x0,y0,numV,ξv,V0)
            end
        end

        fp = open("potential.dat","w")
        for ix=1:Nx
            for iy=1:Ny
                ii = (iy-1)*Nx+ix
                println(fp,ix,"\t",iy,"\t",potential[ix,iy])
            end
            println(fp,"\t")
        end
        close(fp)

        return potential

    end


    function loadfromfile(inputdata,Nx,Ny)
        numofsets = countlines(inputdata)
        data = readlines(inputdata)
        u = split(data[1])
        datai = parse.(Float64,u[:])
        n = length(datai)
        dataset = zeros(Float64,n,Nx*Ny)
        count = 0
        for ix=1:Nx
            for iy=1:Ny
                count += 1
                ii = (iy-1)*Nx+ix
                u = split(data[count])
                datai = parse.(Float64,u[:])
                dataset[:,ii] = datai[:]
            end
            count += 1
        end
        #=
        num = 0
        for inum = 1:numofsets
            u = split(data[inum])
            if length(u) != 0
                datai = parse.(Float64,u[:])
                push!(dataset,datai)
                num += 1
            end
        end
        println("num. of data is $num")
        =#
        return dataset
    end

    function loaddeltapsi!(lattice,inputdir,fname)
        Δfile = inputdir*"delta_"*fname*".dat"
        Φfile = inputdir*"Phi_"*fname*".dat"
        nfile = inputdir*"density_"*fname*".dat"
        Nx = lattice.meshes[1]
        Ny = lattice.meshes[2]
        Δdataset= loadfromfile(Δfile,Nx,Ny)
        Φdataset = loadfromfile(Φfile,Nx,Ny)
        ndataset = loadfromfile(nfile,Nx,Ny)



        #if num != Nx*Ny
        #    println("Error! num should be Nx*Ny, num=$num and Nx*Ny = $(Nx*Ny)")
        #    return
        #end

        for ix=1:Nx
            for iy=1:Ny
                ii = (iy-1)*Nx+ix
                #println(ndataset[3,ii])
                lattice.density[ix,iy] = ndataset[3,ii]
                #println(lattice.density[ix,iy])
                lattice.Φ[ix,iy] =Φdataset[3,ii]
                lattice.Δ[ii] = Δdataset[3,ii]+im*Δdataset[4,ii]
            end
        end

    end




    function calc_kinetic(lattice)
        #Nx = lattice.meshes[1]
        #Ny = lattice.meshes[2]
        factor = -hbar^2/(2*lattice.mass)
        #factor = -1/(2*lattice.mass)
        mat_T = spzeros(Float64,lattice.Vol,lattice.Vol)
        for i=1:lattice.Vol
            ix,iy = index2coordinate(i,lattice)
            #2nd derivative: x direction
            indexofcoffs_2nd = lattice.indexofcoffs_2nd[1,:]
            valueofcoffs_2nd = lattice.valueofcoffs_2nd[1,:]

            #println("ix: $ix iy: $iy")
            jy = iy
            count = 0
            for dx in indexofcoffs_2nd
                count += 1
                jx = ix + dx
                jx,jy = periodic!(jx,jy,lattice)
                j = coordinate2index(jx,jy,lattice)
                mat_T[i,j] +=valueofcoffs_2nd[count]*factor
                #println("jx: $jx iy: $jy, $(mat_T[i,j])")
            end
            #println("xend")
            #2nd derivative: y direction
            indexofcoffs_2nd = lattice.indexofcoffs_2nd[2,:]
            valueofcoffs_2nd = lattice.valueofcoffs_2nd[2,:]
            jx = ix
            count = 0
            for dy in indexofcoffs_2nd
                count += 1
                jy = iy + dy
                jx,jy = periodic!(jx,jy,lattice)
                j = coordinate2index(jx,jy,lattice)
                mat_T[i,j] +=valueofcoffs_2nd[count]*factor
                #println("jx: $jx iy: $jy, $(mat_T[i,j])")
            end


        end
        return mat_T
    end

    function calc_kinetic(lattice,magnetic) #for vortex system
        Nx = lattice.meshes[1]
        Ny = lattice.meshes[2]
        u1 = zeros(Float64,2)
        u2 = zeros(Float64,2)
        R = zeros(Float64,2)
        r0 = zeros(Float64,2)
        u1[1] = Nx
        u1[2] = 0.0
        u2[1] = 0.0
        u2[2] = Ny

        dr = lattice.L./(lattice.meshes)
        mφ = lattice.mφ
        φ0 = π

        L = lattice.L
        hz = mφ*φ0/(L[1]*L[2])
        #Ax = lattice.vec_A[1,:]
        #Ay = lattice.vec_A[2,:]

        factor = -hbar^2/(2*lattice.mass)
        #factor = -1/(2*lattice.mass)
        mat_T = spzeros(ComplexF64,lattice.Vol,lattice.Vol)
        #return mat_T2

        for i=1:lattice.Vol
            ix,iy = index2coordinate(i,lattice)
            #2nd derivative: x direction
            indexofcoffs_2nd = lattice.indexofcoffs_2nd[1,:]
            valueofcoffs_2nd = lattice.valueofcoffs_2nd[1,:]
            rx,ry = calc_r(ix,iy,Nx,Ny)
#            rx,ry = calc_r(ix,iy,dr,L)

            #println("ix: $ix iy: $iy")
            jy = iy
            count = 0
            for dx in indexofcoffs_2nd
                count += 1
                jx = ix + dx
                flag_x = ifelse(jx> Nx,1,ifelse(jx < 1,-1,0))

                jx,jy = periodic!(jx,jy,lattice)
                j = coordinate2index(jx,jy,lattice)
#                θ = dx*Ax*dr[1]/c
#                Θ = -ry*hz/2
                Θ = - dx*mφ*π*ry/(2*Nx*Ny)

                χ = 0.0
                if magnetic && flag_x != 0
                    m = flag_x
                    n = 0
                    R[:] = flag_x*u1[:]
                    if mφ == 2 || mφ == 8
                        χ = -mφ*π*(R[1]*(ry) - R[2]*(rx))/(2*Nx*Ny) - (m-n)*π/2
                    end
                    if mφ == 3
                        χ = -mφ*π*(R[1]*(ry) - R[2]*(rx))/(2*Nx*Ny)
                    end



                end
                mat_T[i,j] +=valueofcoffs_2nd[count]*factor*exp(im*(χ+Θ))
                #println("jx: $jx iy: $jy, $(mat_T[i,j])")
            end
            #println("xend")
            #2nd derivative: y direction
            indexofcoffs_2nd = lattice.indexofcoffs_2nd[2,:]
            valueofcoffs_2nd = lattice.valueofcoffs_2nd[2,:]
            jx = ix
            count = 0
            for dy in indexofcoffs_2nd
                count += 1
                jy = iy + dy
                flag_y = ifelse(jy> Ny,1,ifelse(jy < 1,-1,0))
                jx,jy = periodic!(jx,jy,lattice)
                j = coordinate2index(jx,jy,lattice)
#                θ = dy*Ay*dr[2]/c
                Θ =  dy*mφ*π*rx/(2*Nx*Ny)
                χ = 0.0
                if magnetic && flag_y != 0
                    m = 0
                    n = flag_y
                    R[:] = flag_y*u2[:]
                    if mφ == 2 || mφ == 8
                        χ = -mφ*π*(R[1]*(ry) - R[2]*(rx))/(2*Nx*Ny) - (m-n)*π/2
                    end
                    if mφ == 3
                        χ = -mφ*π*(R[1]*(ry) - R[2]*(rx))/(2*Nx*Ny)
                    end

                end
                mat_T[i,j] +=valueofcoffs_2nd[count]*factor*exp(im*(χ+Θ))
                #println("jx: $jx iy: $jy, $(mat_T[i,j])")
            end


        end



        return mat_T
    end

    function calc_r(ix,iy,Nx,Ny)
        rcx = (Nx-1)/2 #the origin
        rcy = (Ny-1)/2 #the origin
        rx = ix - rcx-1
        ry = iy - rcy-1

        return rx,ry
    end

    function calc_r(ix,iy,dr::Array{Float64,1},L::Array{Float64,1})
        x = (ix-1/2)*dr[1]- L[1]/2
        y = (iy-1/2)*dr[2]- L[2]/2
        return x,y
    end




    function make_Hnormal(lattice)
        if lattice.real
            mat_H = spzeros(Float64,lattice.Vol,lattice.Vol)
        else
            mat_H = spzeros(ComplexF64,lattice.Vol,lattice.Vol)
        end

        if lattice.mφ == 0
            mat_T = calc_kinetic(lattice)
        else
            mat_T = calc_kinetic(lattice,true)
        end

        mat_H += mat_T

        for i=1:lattice.Vol
            ix,iy = index2coordinate(i,lattice)
            mat_H[i,i] += -lattice.μ
            if lattice.use_Φ
#                mat_H[i,i] += -lattice.Φ[ix,iy]
                mat_H[i,i] += lattice.Φ[ix,iy]
            end
            if lattice.nimp != 0
                mat_H[i,i] += lattice.potential[ix,iy]
            end
        end
#        println(mat_H)

        return mat_H
    end

    function make_HBdG(lattice)
        N = lattice.Vol
        mat_Hnormal = make_Hnormal(lattice)

        if lattice.real
            HBdG = spzeros(Float64,2N,2N)
        else
            HBdG = spzeros(ComplexF64,2N,2N)
        end

        HBdG[1:N,1:N] = mat_Hnormal[1:N,1:N]
        HBdG[1+N:2N,1+N:2N] = -conj.(mat_Hnormal[1:N,1:N])

        for i=1:N
            j=i+N
            HBdG[i,j] =lattice.Δ[i]
            HBdG[j,i] =conj(lattice.Δ[i])
        end
        return HBdG

    end

    function update_HBdG!(HBdG,lattice,Φold)
        N = lattice.Vol
        #d = 100000.0

        for i=1:N
            j=i+N
            HBdG[i,j] =lattice.Δ[i]
            HBdG[j,i] =conj(lattice.Δ[i])
        end

        if lattice.use_Φ
            for i=1:N
                ix,iy = index2coordinate(i,lattice)
                HBdG[i,i] += -Φold[ix,iy]+lattice.Φ[ix,iy]
                HBdG[i+N,i+N] = -conj(HBdG[i,i])
            end
        end

        return HBdG

    end

    function calc_current(lattice,mat_H,cdc)
        Nx = lattice.meshes[1]
        Ny = lattice.meshes[2]
        current_x = spzeros(Float64,Nx,Ny)
        current_y = spzeros(Float64,Nx,Ny)


        u1 = zeros(Float64,2)
        u2 = zeros(Float64,2)
        R = zeros(Float64,2)
        r0 = zeros(Float64,2)
        u1[1] = Nx
        u1[2] = 0.0
        u2[1] = 0.0
        u2[2] = Ny
        dr = lattice.L./(lattice.meshes .-1)
        mφ = lattice.mφ

        for i = 1:lattice.Vol
            ix,iy = index2coordinate(i,lattice)
            #println("ix = $ix, iy = $iy")
            rx,ry = calc_r(ix,iy,Nx,Ny)

            dx = 1
            jx = ix + dx
            jy = iy
            flag_x = ifelse(jx> Nx,1,ifelse(jx < 1,-1,0))

            jx,jy = periodic!(jx,jy,lattice)
            j = coordinate2index(jx,jy,lattice)
            Θ = - dx*mφ*π*ry/(2*Nx*Ny)
            χ = 0.0
            if flag_x != 0
                m = flag_x
                n = 0
                R[:] = flag_x*u1[:]
                if mφ == 2
                    χ = -mφ*π*(R[1]*(ry) - R[2]*(rx))/(2*Nx*Ny) - (m-n)*π/2
                end

            end
            #println("jx = $jx, iy = $jy")
            #println(exp(im*(Θ+χ)),"\t",cdc[i,j],"\t",exp(im*(Θ+χ))*cdc[i,j],"\t",exp(im*(Θ))*cdc[i,j])
            current_x[ix,iy] += 2*imag(exp(im*(Θ+χ))*cdc[i,j])

            dx = -1
            jx = ix + dx
            jy = iy
            flag_x = ifelse(jx> Nx,1,ifelse(jx < 1,-1,0))

            jx,jy = periodic!(jx,jy,lattice)
            j = coordinate2index(jx,jy,lattice)
            Θ = - dx*mφ*π*ry/(2*Nx*Ny)
            χ = 0.0
            if flag_x != 0
                m = flag_x
                n = 0
                R[:] = flag_x*u1[:]
                if mφ == 2
                    χ = -mφ*π*(R[1]*(ry) - R[2]*(rx))/(2*Nx*Ny) - (m-n)*π/2
                end
                #println("flag_x = $(flag_x) ", exp(im*(Θ+χ)),"\t",cdc[i,j])
            end
            #println("jx = $jx, iy = $jy")
            #println(exp(im*(Θ+χ)),"\t",cdc[i,j],"\t",exp(im*(Θ+χ))*cdc[i,j],"\t",exp(im*(Θ))*cdc[i,j])
            current_x[ix,iy] += -2*imag(exp(im*(Θ+χ))*cdc[i,j])

            dy = 1
            jy = iy + dy
            jx = ix
            flag_y = ifelse(jy> Ny,1,ifelse(jy < 1,-1,0))
            jx,jy = periodic!(jx,jy,lattice)
            j = coordinate2index(jx,jy,lattice)
            Θ =  dy*mφ*π*rx/(2*Nx*Ny)
            χ = 0.0
            if flag_y != 0
                m = 0
                n = flag_y
                R[:] = flag_y*u2[:]
                if mφ == 2
                    χ = -mφ*π*(R[1]*(ry) - R[2]*(rx))/(2*Nx*Ny) - (m-n)*π/2
                end
                #println("flag_y = $(flag_y) ", exp(im*(Θ+χ)),"\t",cdc[i,j])
            end
            #println("jx = $jx, iy = $jy")
            #println(exp(im*(Θ+χ)),"\t",cdc[i,j],"\t",exp(im*(Θ+χ))*cdc[i,j],"\t",exp(im*(Θ))*cdc[i,j])
            current_y[ix,iy] += 2*imag(exp(im*(Θ+χ))*cdc[i,j])

            dy = -1
            jy = iy + dy
            jx = ix
            flag_y = ifelse(jy> Ny,1,ifelse(jy < 1,-1,0))
            jx,jy = periodic!(jx,jy,lattice)
            j = coordinate2index(jx,jy,lattice)
            Θ =  dy*mφ*π*rx/(2*Nx*Ny)
            χ = 0.0
            if flag_y != 0
                m = 0
                n = flag_y
                R[:] = flag_y*u2[:]
                if mφ == 2
                    χ = -mφ*π*(R[1]*(ry) - R[2]*(rx))/(2*Nx*Ny) - (m-n)*π/2
                end
                #println("flag_y = $(flag_y) ", exp(im*(Θ+χ)),"\t",cdc[i,j])
            end
            #println("jx = $jx, iy = $jy")
            #println(exp(im*(Θ+χ)),"\t",cdc[i,j],"\t",exp(im*(Θ+χ))*cdc[i,j],"\t",exp(im*(Θ))*cdc[i,j])
            current_y[ix,iy] += -2*imag(exp(im*(Θ+χ))*cdc[i,j])




        end

        #stop

        current_x = current_x.*(1/(2*lattice.mass)*(1/(2dr[1])))
        current_y = current_y.*(1/(2*lattice.mass)*(1/(2dr[2])))
        return current_x,current_y
    end

    function solve_CN(source,dr,lattice)
        size_n = size(source)
        Nx = size_n[1]
        Ny = size_n[2]
        A = spzeros(Float64,Nx*Ny,Nx*Ny)
        rho = zeros(Float64,Nx*Ny)
        Phi = zeros(Float64,Nx*Ny)
        mean = sum(source)/(Nx*Ny)
        
        for ix=1:Nx
            for iy=1:Ny
                ii = (iy-1)*Nx+ix       
                rho[ii] = source[ix,iy]    -mean
                Phi[ii] = lattice.Φ[ix,iy]     

                dx = 1
                jx = ix+dx
                jy = iy
                jx += ifelse(jx > Nx,-Nx,0)
                jj = (jy-1)*Nx+jx
                

                A[ii,jj] += 1/dr[1]^2
                


                dx = -1
                jx = ix+dx
                jy = iy
                jx += ifelse(jx < 1,Nx,0)
                jj = (jy-1)*Nx+jx
                
                
                A[ii,jj] += 1/dr[1]^2
                

                dy = 1
                jx = ix
                jy = iy+dy
                jy += ifelse(jy > Ny,-Ny,0)
                jj = (jy-1)*Nx+jx
                
                
                A[ii,jj] += 1/dr[2]^2
                


                dy = -1
                jx = ix
                jy = iy+dy
                jy += ifelse(jy < 1,Ny,0)
                jj = (jy-1)*Nx+jx
                
                
                A[ii,jj] += 1/dr[2]^2
                


                jx = ix
                jy = iy
                jj = ii
                A[ii,jj] += -2/dr[1]^2-2/dr[2]^2



                
            end            
        end

        h = 0.001
        L = -h.*copy(A)./2
        R = -copy(L)
        
        for i = 1:Nx*Ny
            L[i,i] += 1
            R[i,i] += 1
        end
        #LR = inv(L)*R
        #e,v = eigen(LR)
        #println(e)

        #A x + rho

        x1 = L \ rho
        Phi = R*Phi
        x2 = L \ Phi
        
        x = x1.*h + x2

        Φnew = copy(lattice.Φ)

        for ix=1:Nx
            for iy=1:Ny
                ii = (iy-1)*Nx+ix  
                Φnew[ix,iy] = x[ii]
            end
        end
        pold = sum(abs.(lattice.Φ))
        if pold != 0 
            println(sum(abs.(lattice.Φ - (Φnew .-sum(Φnew)/(Nx*Ny))))/pold)
        end
        return  Φnew .-sum(Φnew)/(Nx*Ny)


    end

    function solve_Poi(source,dr) #solve ∇^2 Φ(r) = -ρ(r)
        size_n = size(source)
        Nx = size_n[1]
        Ny = size_n[2]

        #=
        ix0 = div(Nx,4)
        iy0 = div(Nx,4)
        ii0 = (iy0-1)*Nx+ix0

#        mean = sum(source)/(Nx*Ny)
        mean = source[ix0,iy0]



        nabla = spzeros(Float64,Nx*Ny-1,Nx*Ny-1)
        rho = zeros(Float64,Nx*Ny-1)
        for ix=1:Nx
            for iy=1:Ny
                ii = (iy-1)*Nx+ix
                ii += ifelse(ii > ii0,-1,0)


                if ix == ix0 && iy==iy0
                else
                    rho[ii] = source[ix,iy]-mean

                    dx = 1
                    jx = ix+dx
                    jy = iy
                    jx += ifelse(jx > Nx,-Nx,0)
                    jj = (jy-1)*Nx+jx
                    jj += ifelse(jj > ii0,-1,0)
                    if jx == ix0 && jy==iy0
                    else
                        nabla[ii,jj] += 1/dr[1]^2
                    end


                    dx = -1
                    jx = ix+dx
                    jy = iy
                    jx += ifelse(jx < 1,Nx,0)
                    jj = (jy-1)*Nx+jx
                    jj += ifelse(jj > ii0,-1,0)
                    if jx == ix0 && jy==iy0
                    else
                        nabla[ii,jj] += 1/dr[1]^2
                    end

                    dy = 1
                    jx = ix
                    jy = iy+dy
                    jy += ifelse(jy > Ny,-Ny,0)
                    jj = (jy-1)*Nx+jx
                    jj += ifelse(jj > ii0,-1,0)
                    if jx == ix0 && jy==iy0
                    else
                        nabla[ii,jj] += 1/dr[2]^2
                    end


                    dy = -1
                    jx = ix
                    jy = iy+dy
                    jy += ifelse(jy < 1,Ny,0)
                    jj = (jy-1)*Nx+jx
                    jj += ifelse(jj > ii0,-1,0)
                    if jx == ix0 && jy==iy0
                    else
                        nabla[ii,jj] += 1/dr[2]^2
                    end


                    jx = ix
                    jy = iy
                    jj = ii
                    nabla[ii,jj] += -2/dr[1]^2-2/dr[2]^2



                end
            end
        end
        #rho = rho #.- source[ix0,iy0]
        x = nabla \ -rho
        field = zeros(Float64,Nx,Ny)
        for ix=1:Nx
            for iy=1:Ny
                ii = (iy-1)*Nx+ix
                ii += ifelse(ii > ii0,-1,0)


                if ix == ix0 && iy==iy0
                    field[ix,iy] = 0.0
                else
                    field[ix,iy] = x[ii]
                end
            end
        end

        fp = open("testx_"*string(Nx)*".dat","w")
        for ix=1:Nx
            for iy=1:Ny
                x = (ix-1)*dr[1]-Nx*dr[1]/2
                y = (iy-1)*dr[2]-Ny*dr[2]/2
                println(fp,x,"\t",y,"\t",field[ix,iy])
            end
        end
        close(fp)
        return field
        =#


        #mean = sum(source)/(Nx*Ny)

#        source_k =  fft(source.-mean)
        source_k =  fft(source)


        fp = open("testfft_"*string(Nx)*".dat","w")
        for ix=1:Nx
            for iy=1:Ny
                println(fp,2π*(ix-1)/Nx,"\t",2π*(iy-1)/Nx,"\t",real(source_k[ix,iy]),"\t",imag(source_k[ix,iy]))
            end
        end
        close(fp)

        source_ktest = copy(source_k)
        for ix=1:Nx
            for iy=1:Ny
                kx = (ix-1)*2π/Nx
                ky = (iy-1)*2π/Ny
                if sin(kx) > 0.5 && sin(ky) > 0.5
                #if ix > 3*Nx/4 && iy > 3*Ny/4
                else
                    source_ktest[ix,iy] = 0
                end

            end
        end
        source_test =  ifft(source_ktest)
        fp = open("testfft2_"*string(Nx)*".dat","w")
        for ix=1:Nx
            for iy=1:Ny
                println(fp,ix,"\t",iy,"\t",real(source_test[ix,iy]))
            end
        end
        close(fp)


        field_k = zeros(ComplexF64,Nx,Ny)
        Wx = exp(2π*im/Nx)
        Wy = exp(2π*im/Ny)
        cut = 0.000001

        for m=1:Nx
            for n=1:Ny
                #=
                if n==1 && m==1
                    field_k[m,n] = 0.0
                else
                    den = (2*cos((n-1)*2π/Nx) +2*cos((m-1)*2π/Ny)-4 )
                    #if den ==0
                    #    println(m,"\t",n)
                    #end
                    field_k[m,n] = -dr[1]^2*source_k[m,n]/den
                    =#

                #= 
                if m-1==(1)*div(Nx,72) && n-1==(0)*div(Nx,72)
                    #println(source_k[m,n])
                    #println("$Nx $Ny $m $n")
                    #println((m-1)*2π*im/Nx,"\t",(n-1)*2π*im/Ny)
                    #println(- source_k[m,n]/((Wx^(m-1)+Wx^(-(m-1))-2)/dr[1]^2+(Wy^(n-1)+Wy^(-(n-1))-2)/dr[2]^2))
                    kx = (m-1)*2π/Nx
                    ky = (n-1)*2π/Ny
                    den = (exp(im*kx)+exp(-im*kx)+exp(im*ky)+exp(-im*ky)-4)/dr[1]^2
                    den2 = (Wx^(m-1)+Wx^(-(m-1))-2)/dr[1]^2+(Wy^(n-1)+Wy^(-(n-1))-2)/dr[2]^2
                    #println("kx = $kx, ky = $ky, den=$den, den2 = $den2")
                end
                =#
                    #=
                if m != 1 || n != 1
                    kx = (m-1)*2π/Nx
                    ky = (n-1)*2π/Ny
                    den = ((Wx^(m-1)+Wx^(-(m-1))-2)/dr[1]^2+(Wy^(n-1)+Wy^(-(n-1))-2)/dr[2]^2)#exp(im*kx)+exp(-im*kx)+exp(im*ky)+exp(-im*ky)-4
                     #den = kx^2+ky^2
                    field_k[m,n] = - source_k[m,n]/den
                    #/((Wx^(m-1)+Wx^(-(m-1))-2)/dr[1]^2+(Wy^(n-1)+Wy^(-(n-1))-2)/dr[2]^2)
                end
                =#
                kx = (m-1)*2π/Nx
                ky = (n-1)*2π/Ny
                #if kx == 0 && ky == 0
                #else
                if abs(sin(kx)) > cut && abs(sin(ky)) > cut
                #if m > 3*Nx/4 && n > 3*Ny/4
                    kx = (m-1)*2π/Nx
                    ky = (n-1)*2π/Ny
                    den = ((Wx^(m-1)+Wx^(-(m-1))-2)/dr[1]^2+(Wy^(n-1)+Wy^(-(n-1))-2)/dr[2]^2)#exp(im*kx)+exp(-im*kx)+exp(im*ky)+exp(-im*ky)-4
                     #den = kx^2+ky^2
                    field_k[m,n] = - source_k[m,n]/den
                    #/((Wx^(m-1)+Wx^(-(m-1))-2)/dr[1]^2+(Wy^(n-1)+Wy^(-(n-1))-2)/dr[2]^2)
                end
            end
        end


        fp = open("testfk_"*string(Nx)*".dat","w")
        for ix=1:Nx
            for iy=1:Ny
                println(fp,2π*(ix-1)/Nx,"\t",2π*(iy-1)/Nx,"\t",real(field_k[ix,iy]),"\t",imag(field_k[ix,iy]))
            end
        end
        close(fp)



        field =  ifft(field_k)
        return field

    end

    function solve_Poi_A(lattice,current_x,current_y)
        d = lattice.d
        dr = lattice.L ./ (lattice.meshes .- 1)
        
 #       vec_Ax = (1/d)*(4π/c)*real.(solve_CN(current_x,dr,lattice))
 #       vec_Ay = (1/d)*(4π/c)*real.(solve_CN(current_y,dr,lattice))
        vec_Ax = (1/d)*(4π/c)*real.(solve_Poi(current_x,dr))
        vec_Ay = (1/d)*(4π/c)*real.(solve_Poi(current_y,dr))
        return vec_Ax,vec_Ay
    end


    function solve_Poi_rho(lattice)
        d = lattice.d
        dr = lattice.L ./ (lattice.meshes .- 1)
#        vec_ψ = (1/d)*4π*real.(solve_CN(lattice.density,dr,lattice))
        vec_ψ = (1/d)*4π*real.(solve_Poi(lattice.density,dr))
        return vec_ψ
    end


    function solve_gap(lattice,T,ωcut,U,numofloops;mixingratio=0.5,fname="")
        Nx =lattice.meshes[1]
        Ny =lattice.meshes[2]
        Ln = Nx*Ny*2
        A = make_HBdG(lattice)
        eps = 1e-12
        Δold = copy(lattice.Δ)
        Φold = copy(lattice.Φ)
        densityold = copy(lattice.density)
        r = mixingratio
        r2 = 0.3

        if lattice.use_Φ
            i = coordinate2index(div(Nx,4),div(Ny,4),lattice)
            nzero = lattice.density[i] 
            println("nzero = $nzero")
        else
            nzero = 0
        end

        for ite = 1:numofloops

            if lattice.calc_current
                println("RSCG start")
                @time cc,cdc = calc_meanfields_RSCG_both_current(eps,A,Nx,Ny,Ln,T,ωcut)
                println("end")
                current_x,current_y = calc_current(lattice,A,cdc)
                lattice.current[:,:,1] = current_x[:,:]
                lattice.current[:,:,2] = current_y[:,:]
                vec_Ax,vec_Ay = solve_Poi_A(lattice,current_x,current_y)
                lattice.vec_A[:,:,1] = vec_Ax[:,:]
                lattice.vec_A[:,:,2] = vec_Ay[:,:]
            else
                cc,cdc = calc_meanfields_RSCG_both(eps,A,Nx,Ny,Ln,T,ωcut)
            end


            for i=1:lattice.Vol
                lattice.Δ[i] = r*U*cc[i,i]+(1-r)*Δold[i]
                ix,iy = index2coordinate(i,lattice)
                if lattice.nimp == 0
                    i1 = i
                    i2 = coordinate2index(Nx-ix+1,iy,lattice)
                    i3 = coordinate2index(Nx-ix+1,Ny-iy+1,lattice)
                    i4 = coordinate2index(ix,Ny-iy+1,lattice)
                    lattice.density[i] = (cdc[i1,i1]+cdc[i2,i2]+cdc[i3,i3]+cdc[i4,i4])/4
                else
                    lattice.density[i] = cdc[i,i]
                end
            end
            lattice.density *= 1/(lattice.dr[1]*lattice.dr[2])

            fp = open("density_"*fname*".dat","w")
            for ix=1:lattice.meshes[1]
                for iy=1:lattice.meshes[2]
                    ii = (iy-1)*lattice.meshes[1]+ix
                    println(fp,ix,"\t",iy,"\t",real(lattice.density[ii]),"\t",imag(lattice.density[ii]))
                end
                println(fp,"\t")
            end
            close(fp)



            i = coordinate2index(div(Nx,4),div(Ny,4),lattice)
            
            lattice.density = r2*densityold+(1-r2)*lattice.density
            lattice.density = lattice.density .-nzero#  lattice.density[i]
            densityold[:,:] = lattice.density[:,:]
            println("average density: ",sum(lattice.density)/(Nx*Ny))


            lattice.Φ = solve_Poi_rho(lattice)
            lattice.Φ[:,:] = lattice.Φ[:,:] .+ 0.5*sum(lattice.density)/(Nx*Ny)
            lattice.Φ =lattice.Φ*r2 +  (1-r2)*Φold

            #vec_ψ,vec_ψ2 = solve_Poi_rho(lattice)
            #lattice.Φ = vec_ψ2[:,:]

            if ite > 1
                hi = sum(abs.(lattice.Δ-Δold))/sum(abs.(Δold))
                println("Iteration $ite . ratio: $hi, mean value: $(sum(abs.(Δold))/length(Δold))")
            end
            A = make_HBdG(lattice)
            #A = update_HBdG!(A,lattice,Φold)
            Δold = copy(lattice.Δ)
            Φold = copy(lattice.Φ)

            fp = open("delta_"*fname*".dat","w")
            for ix=1:lattice.meshes[1]
                for iy=1:lattice.meshes[2]
                    ii = (iy-1)*lattice.meshes[1]+ix
                    println(fp,ix,"\t",iy,"\t",real(lattice.Δ[ii]),"\t",imag(lattice.Δ[ii]))
                end
                println(fp,"\t")
            end
            close(fp)



            fp = open("Phi_"*fname*".dat","w")
            for ix=1:lattice.meshes[1]
                for iy=1:lattice.meshes[2]
                    ii = (iy-1)*lattice.meshes[1]+ix
                    println(fp,ix,"\t",iy,"\t",lattice.Φ[ix,iy])
                end
                println(fp,"\t")
            end
            close(fp)

            #=
            fp = open("Phi2.dat","w")
            for ix=1:lattice.meshes[1]
                for iy=1:lattice.meshes[2]
                    ii = (iy-1)*lattice.meshes[1]+ix
                    println(fp,ix,"\t",iy,"\t",vec_ψ2[ix,iy])
                end
                println(fp,"\t")
            end
            close(fp)
            =#






            if lattice.calc_current
                fp = open("current_"*fname*".dat","w")
                for ix=1:lattice.meshes[1]
                    for iy=1:lattice.meshes[2]
                        ii = (iy-1)*lattice.meshes[1]+ix
                        println(fp,ix,"\t",iy,"\t",lattice.current[ix,iy,1],"\t",lattice.current[ix,iy,2])
                    end
                    println(fp,"\t")
                end
                close(fp)

                fp = open("vectorA_"*fname*".dat","w")
                for ix=1:lattice.meshes[1]
                    for iy=1:lattice.meshes[2]
                        ii = (iy-1)*lattice.meshes[1]+ix
                        println(fp,ix,"\t",iy,"\t",vec_Ax[ix,iy],"\t",vec_Ay[ix,iy])
                    end
                    println(fp,"\t")
                end
                close(fp)
            end

            #stop
            flush(stdout)
        end

        return lattice.Δ
    end



    #include("./src/coordinates.jl") #periodic etc.
    #include("./src/coefficients.jl") #call_coefficients_2nd etc.




    function periodic!(ix,iy,lattice)
        ix += ifelse(ix > lattice.meshes[1],-lattice.meshes[1],0)
        iy += ifelse(iy > lattice.meshes[2],-lattice.meshes[2],0)
        ix += ifelse(ix < 1,lattice.meshes[1],0)
        iy += ifelse(iy < 1,lattice.meshes[2],0)
        return ix,iy
    end



    function coordinate2index(ix,iy,lattice)
        return (iy-1)*lattice.meshes[1] + ix
    end

    function index2coordinate(i,lattice)
        Nx = lattice.meshes[1]
        Ny = lattice.meshes[2]
        ix = (i-1) % Nx+1
        iy = div(i-ix,Nx)+1
        return ix,iy
    end

    function call_coefficients_2nd(dr,order) #Set the coefficients of 2nd order derivatives
        if order == 2
            indexofcoffs = zeros(Int64,2,3)
            valueofcoffs = zeros(Float64,2,3)

            indexofcoffs[:,1] .= -1
            indexofcoffs[:,2] .= 0
            indexofcoffs[:,3] .= 1
            for i=1:dim
                valueofcoffs[i,1] = 1/dr[i]^2
                valueofcoffs[i,2] = -2/dr[i]^2
                valueofcoffs[i,3] = 1/dr[i]^2
            end

        elseif order ==4
            indexofcoffs = zeros(Int64,2,5)
            valueofcoffs = zeros(Float64,2,5)

            indexofcoffs[:,1] .= -2
            indexofcoffs[:,2] .= -1
            indexofcoffs[:,3] .= 0
            indexofcoffs[:,4] .= 1
            indexofcoffs[:,5] .= 2
            for i=1:dim
                valueofcoffs[i,1] = -1/(12*dr[i]^2)
                valueofcoffs[i,2] = 16/(12*dr[i]^2)
                valueofcoffs[i,3] = -30/(12*dr[i]^2)
                valueofcoffs[i,4] = 16/(12*dr[i]^2)
                valueofcoffs[i,5] = -1/(12*dr[i]^2)
            end
        else
            println("Error!: order should be 2 or 4. Now, order = ",order)
            exit()
        end

        return indexofcoffs,valueofcoffs

    end

    function call_coefficients_1st(dr,order) #Set the coefficients of 2nd order derivatives
        if order == 2
            indexofcoffs = zeros(Int64,2,2)
            valueofcoffs = zeros(Float64,2,2)

            indexofcoffs[:,1] .= -1
            indexofcoffs[:,2] .= 1
            for i=1:dim
                valueofcoffs[i,1] = -1/2dr[i]
                valueofcoffs[i,2] = 1/2dr[i]
            end

        elseif order ==4
            indexofcoffs = zeros(Int64,2,4)
            valueofcoffs = zeros(Float64,2,4)

            indexofcoffs[:,1] .= -2
            indexofcoffs[:,2] .= -1
            indexofcoffs[:,3] .= 1
            indexofcoffs[:,4] .= 2
            for i=1:dim
                valueofcoffs[i,1] = 1/(12dr[i])
                valueofcoffs[i,2] = -8/(12dr[i])
                valueofcoffs[i,3] = 8/(12dr[i])
                valueofcoffs[i,4] = -1/(12dr[i])
            end
        else
            println("Error!: order should be 2 or 4. Now, order = ",order)
            exit()
        end

        return indexofcoffs,valueofcoffs

    end

    function test_2()
        println("num of workers: $(nworkers())")
        dr = 20.0/48
        meshes = [48,48]
        L = meshes*dr
        use_Φ=false
        d = 20.0
        lattice = set_Lattice(meshes,L,real=false,mφ=2,calc_current=true,use_Φ=use_Φ,d = d,nimp=0)


    end

    function test()
        println("num of workers: $(nworkers())")
        #L = [20.0,20.0]
        #meshes = [48,48]
        dr = 20.0/48
#        L = [10.0,10.0]*20/48
        meshes = [96,96]
        L = meshes*dr
        use_Φ=true
        d = 20.0
        lattice = set_Lattice(meshes,L,real=false,mφ=2,calc_current=true,use_Φ=use_Φ,d = d,nimp=0)
        #mat_H = make_Hnormal(lattice)

        inputdir = "./BdG/96x96U095_20o48_w_d20/"
        fname = "w96_4h_d20_dr20o48"

        loaddeltapsi!(lattice,inputdir,fname)
#        return

        @time mat_H = make_HBdG(lattice)
        Nx = lattice.meshes[1]
        Ny = lattice.meshes[2]

        for i=1:Nx*Ny*2
            for j=i:Nx*Ny*2
                v = mat_H[i,j] - conj(mat_H[j,i])
                if v != 0.0
                    println("$i, $j, $v")
                end
            end
        end


        fp = open("delta_pre.dat","w")
        for ix=1:lattice.meshes[1]
            for iy=1:lattice.meshes[2]
                ii = (iy-1)*lattice.meshes[1]+ix
                println(fp,ix,"\t",iy,"\t",real(lattice.Δ[ii]),"\t",imag(lattice.Δ[ii]))
            end
            println(fp,"\t")
        end
        close(fp)


        #return



#        e,v = eigs(mat_H,nev=10)
        numene = 100
        @time e,v = eigs(mat_H,nev=numene,which=:SM)
        #ρ = 0.2
        #γ = 0.0
        #@time e,residuals,v,num = SS.eigs(mat_H,γ,ρ)
        println(e)
        es = sort(real.(e))
        fp = open("eigenvalues.dat","w")
        for i=1:num
            println(fp,es[i])
        end
        close(fp)

        fp = open("wavefunctions0.dat","w")
        for ix=1:lattice.meshes[1]
            for iy=1:lattice.meshes[2]
                ii = (iy-1)*lattice.meshes[1]+ix
                println(fp,ix,"\t",iy,"\t",real(v[ii,2]),"\t",imag(v[ii,2]))
            end
            println(fp,"\t")
        end
        close(fp)
        println(e)

        return sort(real.(e))

        T = 0.01
        ωcut = 30.0
        U = -10.0
        itemax = 200
        @time Δ = solve_gap(lattice,T,ωcut,U,itemax,fname="wo48_4h_d2")
#        e = eigen(Matrix(mat_H)).values



        return Δ
    end

end # module
