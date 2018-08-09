using Distributed
@everywhere module Bdgeq
    
    include("./ChebyshevPolynomial.jl")
    using SparseArrays
    using LinearAlgebra
    using .Chebyshev
    #import Plots
    export iteration,calc_A



    function calc_A_vortex(Nx,Ny,μ,Δ,aa)
        Ln = Nx*Ny*2
        A = spzeros(Complex{Float64},Ln,Ln)
        u1 = zeros(Float64,2)
        u2 = zeros(Float64,2)
        R = zeros(Float64,2)
        r0 = zeros(Float64,2) 
        u1[1] = Nx
        u1[2] = 0.0
        u2[1] = 0.0
        u2[2] = Ny
        
        

    
        for ix=1:Nx
            for iy=1:Ny
                periodic_x = 0
                periodic_y = 0
            
                rx,ry = calc_r(ix,iy,Nx,Ny)
            
                #Diagonal element
                ii = (iy-1)*Nx+ix
                jx = ix                        
                jy = iy
                jj = (jy-1)*Nx+jx
                A[ii,jj] = -μ
                χ = 0.0
            
                #+1 in x direction
                jx = ifelse(ix == Nx,1,ix+1)
                periodic_x = ifelse(ix == Nx,+1,0)            
                jy = iy            
                Θ = - π*ry/(2*Nx*Ny) 
                if periodic_x == 1
                    m = 1
                    n = 0
                    R[:] = u1[:]  
                    
                    χ = - π*(R[1]*(ry+2r0[2]) - R[2]*(rx+2r0[1]))/(2*Nx*Ny)
                end
              
                jj = (jy-1)*Nx+jx
                A[ii,jj] = -1.0*exp(im*(Θ+χ))
            
            
                #-1 in x direction
                χ = 0.0
                jx = ifelse(ix == 1,Nx,ix-1)
                periodic_x = ifelse(ix == 1,-1,0)  
                Θ =  π*ry/(2*Nx*Ny)
              
                jy = iy
                if periodic_x == -1
                    m = -1
                    n = 0
                    R[:] = -u1[:] 
                    
                    χ = - π*(R[1]*(ry+2r0[2]) - R[2]*(rx+2r0[1]))/(2*Nx*Ny) 
                end
            
                jj = (jy-1)*Nx+jx
                A[ii,jj] = -1.0*exp(im*(Θ+χ))
                periodic_x = 0
            
            
            
                #+1 in y direction
                χ = 0.0
                jx = ix
                jy = ifelse(iy == Ny,1,iy+1)
                periodic_y = ifelse(iy == Ny,+1,0)
                Θ =  π*rx/(2*Nx*Ny)
                if periodic_y == 1
                    m = 0
                    n = 1
                    R[:] = u2[:]  
                    χ = - π*(R[1]*(ry+2r0[2]) - R[2]*(rx+2r0[1]))/(2*Nx*Ny) 
                end
                jj = (jy-1)*Nx+jx
                A[ii,jj] = -1.0*exp(im*(Θ+χ))
            
            
                #-1 in y direction
                χ = 0.0
                jx = ix
                jy = ifelse(iy == 1,Ny,iy-1)
                periodic_y = ifelse(iy == 1,-1,0)
                Θ = - π*rx/(2*Nx*Ny)
                if periodic_y == -1
                    m = 0
                    n = -1
                    R[:] = -u2[:] 
                    χ = - π*(R[1]*(ry+2r0[2]) - R[2]*(rx+2r0[1]))/(2*Nx*Ny)                  
                end            
                jj = (jy-1)*Nx+jx
                A[ii,jj] = -1.0*exp(im*(Θ+χ))           
                            
            end
        end
    
        for ii=1:Nx*Ny
            for jj=1:Nx*Ny
                A[ii+Nx*Ny,jj+Nx*Ny] = -conj(A[ii,jj])
                #A[ii,jj+Nx*Ny] = Δ[ii,jj]
                #A[ii+Nx*Ny,jj] = conj(Δ[jj,ii])
            end
        end

        for ii = 1:Nx*Ny
            A[ii,ii+Nx*Ny] = Δ[ii,ii]
            A[ii+Nx*Ny,ii] = conj(Δ[ii,ii])
        end
    
        return A/aa
    
    end

    function calc_r(ix,iy,Nx,Ny)
        rcx = (Nx-1)/2 #the origin 
        rcy = (Ny-1)/2 #the origin
        rx = ix - rcx-1
        ry = iy - rcy-1
           
        return rx,ry
    end


    function calc_A(Nx,Ny,μ,Δ,aa)
        Ln = Nx*Ny*2
        A = spzeros(Ln,Ln)
    
        for ix=1:Nx
            for iy=1:Ny
                #Diagonal element
                ii = (iy-1)*Nx+ix
                jx = ix
                jy = iy
                jj = (jy-1)*Nx+jx
                A[ii,jj] = -μ
                #+1 in x direction
                jx = ifelse(ix == Nx,1,ix+1)
                jy = iy
                jj = (jy-1)*Nx+jx
                A[ii,jj] = -1.0
                #-1 in x direction
                jx = ifelse(ix == 1,Nx,ix-1)
                jy = iy
                jj = (jy-1)*Nx+jx
                A[ii,jj] = -1.0
                #+1 in y direction
                jx = ix
                jy = ifelse(iy == Ny,1,iy+1)
                jj = (jy-1)*Nx+jx
                A[ii,jj] = -1.0
                #-1 in y direction
                jx = ix
                jy = ifelse(iy == 1,Ny,iy-1)
                jj = (jy-1)*Nx+jx
                A[ii,jj] = -1.0            
                            
            end
        end
    
        for ii=1:Nx*Ny
            for jj=1:Nx*Ny
                A[ii+Nx*Ny,jj+Nx*Ny] = -conj(A[ii,jj])
                A[ii,jj+Nx*Ny] = Δ[ii,jj]
                A[ii+Nx*Ny,jj] = conj(Δ[jj,ii])
            end
        end
    
        return A/aa
    
    end


    function update_A!(A,Nx,Ny,μ,Δ,aa)
        for ii=1:Nx*Ny
            for jj=1:Nx*Ny
                A[ii,jj+Nx*Ny] = Δ[ii,jj]/aa
                A[ii+Nx*Ny,jj] = conj(Δ[jj,ii])/aa
            end
        end
    end


    function iteration(nc,Nx,Ny,aa,bb,ωc,U,initialΔ,μ,full,vortex,itemax)

        if vortex
            Δ = sparse((initialΔ+0*im)*I, Nx*Ny, Nx*Ny)
        
            #speye(Nx*Ny,Nx*Ny)*(initialΔ+0*im)
            Δold = sparse((initialΔ+0*im)*I, Nx*Ny, Nx*Ny) #speye(Nx*Ny,Nx*Ny)*(initialΔ+0*im)
            A = calc_A_vortex(Nx,Ny,μ,Δ,aa)
        else   
            Δ = sparse((initialΔ+0*im)*I, Nx*Ny, Nx*Ny)#speye(Nx*Ny,Nx*Ny)*initialΔ
            Δold = sparse((initialΔ+0*im)*I, Nx*Ny, Nx*Ny) #speye(Nx*Ny,Nx*Ny)*initialΔ     
            A = calc_A(Nx,Ny,μ,Δ,aa)        
        end
        mat_Δ = zeros(typeof(Δ[1,1]),Nx,Ny)    
        
        for ite=1:itemax
            
            if full
               @time Δ = calc_meanfields(A,Nx,Ny,ωc) 
            else
               @time Δ = calc_meanfields(nc,A,Nx,Ny,aa,bb,ωc)
            end
           
            
           
            Δ = Δ*U
            update_A!(A,Nx,Ny,μ,Δ,aa)
        
            eps = 0.0
            nor = 0.0
            for i=1:Nx*Ny
                eps += abs(Δ[i,i]-Δold[i,i])^2
                nor += abs(Δold[i,i])^2
            end
            eps = eps/nor
            println("ite = ",ite," eps = ",eps)
            if eps <= 1e-6
                println("End ",Δ[div(Nx,2),div(Ny,2)])
                break
            end
            Δold = Δ
        
            fp = open("./gap.dat","w")
            for ix=1:Nx
                for iy=1:Ny
                    ii = (iy-1)*Nx + ix
                    mat_Δ[ix,iy] = Δ[ii,ii]
                    println(fp,ix,"\t",iy,"\t",abs(mat_Δ[ix,iy]),"\t",angle(mat_Δ[ix,iy]))  
                end
                 println(fp," ")
            end
            println("center (Nx/2,Ny/2): ",mat_Δ[div(Nx,2),div(Ny,2)])
            println("corner (1,1): ",mat_Δ[1,1])
            #plot(mat_Δ)
        
             
            close(fp)
        
        
        end


    
        return mat_Δ
    
    end

end