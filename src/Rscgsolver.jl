@everywhere module Rscgsolver
    export calc_meanfields_RSCG_both,calc_meanfields_RSCG_both_current
    using SparseArrays
    using LinearAlgebra
    using Distributed
    using RSCG

    function RSCGs(eps,n_omega,left_i,right_j,vec_sigma,A,Ln)

    #--Line 2 in Table III.
        vec_x = zeros(Float64,Ln)
        vec_b =   zeros(Float64,Ln)
        vec_b[right_j] = 1.0

        vec_r = zeros(Ln)
        vec_p = zeros(Ln)
        vec_r[right_j] = 1.0
        vec_p[right_j] = 1.0
        alpham = 1.0
        betam = 0.0
    #--
        Sigma = vec_b[left_i] #Line 3. Sigma=V.b. V=v1^T, v1^T=e(j)^T = (0,0,0,0,...,1,...,0,0,0)

        vec_Ap = zeros(Ln)
        vec_g = zeros(Complex{Float64},n_omega)
        vec_rhok = ones(Complex{Float64},n_omega)
        vec_rhokp = ones(Complex{Float64},n_omega)
        vec_rhokm = ones(Complex{Float64},n_omega)
        vec_alpha = zeros(Complex{Float64},n_omega)
        vec_beta = zeros(Complex{Float64},n_omega)
        vec_Theta = zeros(Complex{Float64},n_omega)
        vec_Pi = ones(Complex{Float64},n_omega)*Sigma
        #---
        #flag = true
        hi = 1.0

        ep = 1e-15

        for ite in 1:200
        #while abs(hi) > eps
            #vec_Ap =
            mul!(vec_Ap, A, -vec_p)
#            A_mul_B!(vec_Ap,A,-vec_p)
            rsum = vec_r'*vec_r#  dot(vec_r,vec_r) #(rk,rk)
            pAp = vec_p'*vec_Ap
            alpha = rsum/pAp #np.dot(vec_p,vec_Ap) #Line 6 (rk,rk)/(pk,A pk)
#        print alpha,rsum

            for i=1:Ln
                vec_x[i] += alpha*vec_p[i] #Line 7
                vec_r[i] += -alpha*vec_Ap[i]#Line 8
            end
#            vec_x += alpha*vec_p #Line 7
#            vec_r += - alpha*vec_Ap #Line 8
            beta = vec_r'*vec_r/rsum #Line9 (r_{k+1},r_{k+1})/(rk,rk)
            vec_p = vec_r + beta*vec_p #Line 10
            Sigma = vec_r[left_i] #Line 11 Sigma=V.r_{k+1}



            #---- Lines 12-17
            for j in 1:n_omega
                update = ifelse(abs(vec_rhok[j]) > eps,true,false)
                if update
                    vec_rhokp[j] = vec_rhok[j]*vec_rhokm[j]*alpham/(vec_rhokm[j]*alpham*(1.0+alpha*vec_sigma[j])+alpha*betam*(vec_rhokm[j]-vec_rhok[j]))#Line 13
                    vec_alpha[j] = alpha*vec_rhokp[j]/vec_rhok[j]#Line 14
                    vec_Theta[j] += vec_alpha[j]*vec_Pi[j] #Line 15
                    vec_beta[j] = ((vec_rhokp[j]/vec_rhok[j])^2)*beta #Line 16
                    vec_Pi[j] = vec_rhokp[j]*Sigma+ vec_beta[j]*vec_Pi[j] #Line 17
                end

                vec_g[j] = vec_Theta[j]
                vec_rhokm[j] = vec_rhok[j]
                vec_rhok[j] = vec_rhokp[j]

            end


            #----
            alpham = alpha
            betam = beta
            hi = rsum


        end

        return vec_g
    end

    function RSCG_vec(eps,vec_left,right_j,σ,A,n)
        N=length(σ)
        atype = eltype(A)
        b = zeros(atype,n)
        b[right_j] = 1.0

        m = length(vec_left)
        kmax = 20000



    #--Line 2 in Table III.
        x = zeros(atype,n)
        r = copy(b)
        p = copy(b)
        αm = 1.0
        βm = 0.0
    #--
        Σ =zeros(atype,m)
        for mm=1:m
            Σ[mm] = b[vec_left[mm]] #Line 3
        end
        Θ = zeros(Complex{Float64},N,m)
        Π = ones(Complex{Float64},N,m)
        for mm=1:m
            Π[:,mm] *= Σ[mm]
        end
        ρk = ones(Complex{Float64},N)
        ρkm = copy(ρk)
        ρkp = copy(ρk)
        Ap = similar(p)

        for k=0:kmax
            mul!(Ap,A,-p)
            #A_mul_B!(Ap,A,-p)
            rnorm = r'*r
            α = rnorm/(p'*Ap)
            x += α*p #Line 7
            r += -α*Ap #Line 8
            β = r'*r/rnorm #Line9
            p = r + β*p #Line 10

            for mm=1:m
                Σ[mm] = r[vec_left[mm]] #Line 11
            end

            for j = 1:N
                update = ifelse(abs(ρk[j]) > eps,true,false)
                if update
                    ρkp[j] = ρk[j]*ρkm[j]*αm/(ρkm[j]*αm*(1.0+α*σ[j])+α*βm*(ρkm[j]-ρk[j]))#Line 13
                    αkj = α*ρkp[j]/ρk[j]#Line 14
                    Θ[j,:] += αkj*Π[j,:] #Line 15
                    βkj = ((ρkp[j]/ρk[j])^2)*β #Line 16
                    Π[j,:] = ρkp[j]*Σ+ βkj*Π[j,:] #Line 17

                end
                ρkm[j] = ρk[j]
                ρk[j] = ρkp[j]
            end
            αm = α
            βm = β
            hi = real(rnorm)
            if hi < eps
                return Θ
            end
        end


        println("Not converged")
        return Θ

    end



    function calc_meanfields_full_finite(A,Nx,Ny,Ln,T,omegamax)

        cc = spzeros(Nx*Ny,Nx*Ny) #lil_matrix((Nx*Ny,Nx*Ny))
#    omegamax = omegac #pi*T(2*n+1), omegac/(T*pi)
        n_omega = Int((Int(omegamax/(T*pi)))/2-1)

        vec_sigma = zeros(Complex{Float64},2*n_omega)

        for n=1:2*n_omega
            vec_sigma[n] = π*T*(2.0*(n-n_omega-1)+1)*im
        end
        A = Matrix(A)
        w,v = eigen(A)


        for ix= 1:Nx#  in range(Nx):
            for iy= 1:Ny#  in range(Ny):
                ii = (iy-1)*Nx+ix
                jj = ii + Nx*Ny
                right_j = jj
                left_i = ii
                vec_g = calc_green(w,v,2*n_omega,left_i,right_j,vec_sigma,A,Ln)
                vec_g += -1 ./(vec_sigma.*vec_sigma)
                cc[ii,ii] = real(T*sum(vec_g))-1/(T*4)

                #stop
            end
        end

        return cc
    end

    function calc_meanfields_full_finite_HF(A,Nx,Ny,Ln,T,omegamax)

        cdc = spzeros(Nx*Ny,Nx*Ny) #lil_matrix((Nx*Ny,Nx*Ny))
#    omegamax = omegac #pi*T(2*n+1), omegac/(T*pi)
        n_omega = Int((Int(omegamax/(T*pi)))/2-1)

        vec_sigma = zeros(Complex{Float64},2*n_omega)

        for n=1:2*n_omega
            vec_sigma[n] = π*T*(2.0*(n-n_omega-1)+1)*im
        end
        A = Matrix(A)
        w,v = eigen(A)


        for ix= 1:Nx#  in range(Nx):
            for iy= 1:Ny#  in range(Ny):
                ii = (iy-1)*Nx+ix
                jj = ii# + Nx*Ny
                right_j = jj
                left_i = ii
                vec_g = calc_green(w,v,2*n_omega,left_i,right_j,vec_sigma,A,Ln)
                vec_g += -1 ./vec_sigma

                cdc[ii,ii] = real(T*sum(vec_g))+1/2
            end
        end

        return cdc
    end

    function calc_green(w,v,n_omega,left_i,right_j,vec_sigma,A,Ln)
        vec_g = zeros(Complex{Float64},n_omega)

        for i=1:Ln
            for n=1:n_omega
                vec_g[n] += v[left_i,i]*v[right_j,i]/(vec_sigma[n] - w[i])
            end
        end
        return vec_g
    end

    function calc_meanfields_RSCG_HF(eps,A,Nx,Ny,Ln,T,omegamax)

        cc = spzeros(Nx*Ny,Nx*Ny) #lil_matrix((Nx*Ny,Nx*Ny))
#    omegamax = omegac #pi*T(2*n+1), omegac/(T*pi)
        n_omega = Int((Int(omegamax/(T*pi)))/2-1)

        vec_sigma = zeros(Complex{Float64},2*n_omega)
#        shift = 3.0
#        for i=1:Ln
#            A[i,i] = A[i,i] + shift
#        end


        for n=1:2*n_omega# in range(2*n_omega):
            #println(π*T*(2.0*(n-n_omega-1)+1)*im)
            vec_sigma[n] = π*T*(2.0*(n-n_omega-1)+1)*im #+shift
        end


        for ix= 1:Nx#  in range(Nx):
            for iy= 1:Ny#  in range(Ny):
                ii = (iy-1)*Nx+ix
                jj = ii #+ Nx*Ny
                right_j = jj#+ Nx*Ny
                left_i = ii#+ Nx*Ny
                vec_g = RSCG(eps,2*n_omega,left_i,right_j,vec_sigma,A,Ln)
                vec_g += -1 ./vec_sigma
                cc[ii,ii] = real(T*sum(vec_g))+1/2
            end
        end

        return cc
    end


    function calc_meanfields_RSCG(eps,A,Nx,Ny,Ln,T,omegamax)

        cc = spzeros(Nx*Ny,Nx*Ny) #lil_matrix((Nx*Ny,Nx*Ny))
#    omegamax = omegac #pi*T(2*n+1), omegac/(T*pi)
        n_omega = Int((Int(omegamax/(T*pi)))/2-1)

        vec_sigma = zeros(Complex{Float64},2*n_omega)


        for n=1:2*n_omega# in range(2*n_omega):
            #println(π*T*(2.0*(n-n_omega-1)+1)*im)
            vec_sigma[n] = π*T*(2.0*(n-n_omega-1)+1)*im#+shift
        end

        vec_c = pmap(ii -> calc_meanfields_RSCG_i(ii,vec_sigma,A,Ln,T,Nx,Ny,eps),1:Nx*Ny)
        for ii=1:Nx*Ny
            cc[ii,ii] = vec_c[ii]
            #calc_meanfields_RSCG_i(ii,vec_sigma,A,Ln,T,Nx,Ny,eps)
        end

        #=
        for ix= 1:Nx#  in range(Nx):
            for iy= 1:Ny#  in range(Ny):
                ii = (iy-1)*Nx+ix
                jj = ii + Nx*Ny
                right_j = jj
                left_i = ii
                vec_left = [left_i]
                vec_g = RSCG_vec(eps,vec_left,right_j,vec_sigma,A,Ln)[:,1]

#                vec_g = RSCG(eps,2*n_omega,left_i,right_j,vec_sigma,A,Ln)
                vec_g += -1 ./(vec_sigma.*vec_sigma)
                cc[ii,ii] = real(T*sum(vec_g))-1/(T*4)
            end
        end
    =#

        return cc
    end

    function calc_meanfields_RSCG_i(ii,vec_sigma,A,Ln,T,Nx,Ny,eps)
        jj = ii + Nx*Ny
        right_j = jj
        left_i = ii
        vec_left = [left_i]
        vec_g = RSCG_vec(eps,vec_left,right_j,vec_sigma,A,Ln)[:,1]

#                vec_g = RSCG(eps,2*n_omega,left_i,right_j,vec_sigma,A,Ln)
        vec_g += -1 ./(vec_sigma.*vec_sigma)
        return real(T*sum(vec_g))-1/(T*4)
    end



    function calc_meanfields_RSCG_both_current(eps,A,Nx,Ny,Ln,T,omegamax)
        cc = spzeros(Complex{Float64},Nx*Ny,Nx*Ny) #lil_matrix((Nx*Ny,Nx*Ny))
        cdc = spzeros(Complex{Float64},Nx*Ny,Nx*Ny)
#    omegamax = omegac #pi*T(2*n+1), omegac/(T*pi)
        #println(round(Int,omegamax/(T*π)))
        n_omega = round(Int,(omegamax/(T*π))/2-1)

        vec_sigma = zeros(Complex{Float64},2*n_omega)

        for n=1:2*n_omega# in range(2*n_omega):
            #println(π*T*(2.0*(n-n_omega-1)+1)*im)
            vec_sigma[n] = π*T*(2.0*(n-n_omega-1)+1)*im#+shift
        end

        @time vec_cs = pmap(ii -> calc_meanfields_RSCG_both_current_i(ii,vec_sigma,A,Ln,T,Nx,Ny,eps),1:Nx*Ny)

        for ii=1:Nx*Ny
            cc_i,cdc_i,c1,c2,c3,c4 = vec_cs[ii]#calc_meanfields_RSCG_both_i(ii,vec_sigma,A,Ln,T,Nx,Ny,eps)
            cc[ii,ii] = cc_i
            cdc[ii,ii] = cdc_i

            ix = (ii-1) % Nx+1
            iy = div(ii-ix,Nx)+1

            jx = ix + 1
            jy = iy
            jx += ifelse(jx > Nx,-Nx,0)
            j1 =  (jy-1)*Nx + jx

            jx = ix -1
            jy = iy
            jx += ifelse(jx < 1,Nx,0)
            j2 =  (jy-1)*Nx + jx

            jx = ix
            jy = iy +1
            jy += ifelse(jy > Ny,-Ny,0)
            j3 =  (jy-1)*Nx + jx

            jx = ix
            jy = iy -1
            jy += ifelse(jy <1,Ny,0)
            j4 =  (jy-1)*Nx + jx
            #println("$j1, $j2, $j3, $j4")
            #println("ii=$ii, js = $js")

            cdc[j1,ii] = c1
            cdc[j2,ii] = c2
            cdc[j3,ii] = c3
            cdc[j4,ii] = c4
            cdc[ii,j1] = conj(c1)
            cdc[ii,j2] = conj(c2)
            cdc[ii,j3] = conj(c3)
            cdc[ii,j4] = conj(c4)
        end

        return cc,cdc
    end

    function calc_meanfields_RSCG_both(eps,A,Nx,Ny,Ln,T,omegamax)
        cc = spzeros(Complex{Float64},Nx*Ny,Nx*Ny) #lil_matrix((Nx*Ny,Nx*Ny))
        cdc = spzeros(Complex{Float64},Nx*Ny,Nx*Ny)
#    omegamax = omegac #pi*T(2*n+1), omegac/(T*pi)
        #println(round(Int,omegamax/(T*π)))
        n_omega = round(Int,(omegamax/(T*π))/2-1)

        vec_sigma = zeros(Complex{Float64},2*n_omega)

        for n=1:2*n_omega# in range(2*n_omega):
            #println(π*T*(2.0*(n-n_omega-1)+1)*im)
            vec_sigma[n] = π*T*(2.0*(n-n_omega-1)+1)*im#+shift
        end

        vec_cs = pmap(ii -> calc_meanfields_RSCG_both_i(ii,vec_sigma,A,Ln,T,Nx,Ny,eps),1:Nx*Ny)

        for ii=1:Nx*Ny
            cc_i,cdc_i = vec_cs[ii]#calc_meanfields_RSCG_both_i(ii,vec_sigma,A,Ln,T,Nx,Ny,eps)
            cc[ii,ii] = cc_i
            cdc[ii,ii] = cdc_i
        end

        return cc,cdc
    end



    function calc_meanfields_RSCG_both_i(ii,vec_sigma,A,Ln,T,Nx,Ny,eps)
        jj = ii + Nx*Ny
        right_j = jj
        left_i = ii

        vec_left = [ii,ii+Nx*Ny] #<c_i c_i> <c_i c_i^+>
        vec_g = RSCG_vec(eps,vec_left,right_j,vec_sigma,A,Ln)

        #vec_g[:,1] += -1 ./(vec_sigma.*vec_sigma)
        vec_g[:,2] += -1 ./(vec_sigma)
        cc_i = T*sum(vec_g[:,1])#-1/(T*4)
        cdc_i = real(T*sum(vec_g[:,2]))+1/2
        cdc_i = 1-cdc_i
        return cc_i,cdc_i
    end

    function calc_meanfields_RSCG_both_current_i(ii,vec_sigma,A,Ln,T,Nx,Ny,eps)
        jj = ii + Nx*Ny
        right_j = jj
        left_i = ii

        ix = (ii-1) % Nx+1
        iy = div(ii-ix,Nx)+1

        jx = ix + 1
        jy = iy
        jx += ifelse(jx > Nx,-Nx,0)
        j1 =  (jy-1)*Nx + jx

        jx = ix -1
        jy = iy
        jx += ifelse(jx < 1,Nx,0)
        j2 =  (jy-1)*Nx + jx

        jx = ix
        jy = iy +1
        jy += ifelse(jy > Ny,-Ny,0)
        j3 =  (jy-1)*Nx + jx
        jx = ix
        jy = iy -1
        jy += ifelse(jy <1,Ny,0)
        j4 =  (jy-1)*Nx + jx

        vec_left = [ii,ii+Nx*Ny,j1+ Nx*Ny,j2+ Nx*Ny,j3+ Nx*Ny,j4+ Nx*Ny] #<c_i c_i> <c_i c_i^+> <c_i c_j^+>

        vec_g = greensfunctions_fortran(vec_left,right_j,A,vec_sigma,eps=eps)

#        vec_g = greensfunctions(vec_left,right_j,vec_sigma,A,eps=eps)
#        vec_g = RSCG_vec(eps,vec_left,right_j,vec_sigma,A,Ln)

        #vec_g[:,1] += -1 ./(vec_sigma.*vec_sigma)
        vec_g[:,2] += -1 ./(vec_sigma)
        cc_i = T*sum(vec_g[:,1])#-1/(T*4)
        cdc_i = real(T*sum(vec_g[:,2]))+1/2
        cdc_i = 1-cdc_i
        #<c_j^+ c_i> = -<c_i c_j^+> if i != j
        #We have to consider cdc[j,i] = <c_j^+ c_i>


        return cc_i,cdc_i,-T*sum(vec_g[:,3]),-T*sum(vec_g[:,4]),-T*sum(vec_g[:,5]),-T*sum(vec_g[:,6])
    end




    function greensfunctions_fortran(i,j,A,σ)
        n = size(A,1)
        M = length(σ)
        Theta = zeros(ComplexF64,M)
        col = A.colptr
        val = A.nzval
        row = A.rowval
        val_l = length(val)
        row_l = length(row)
        col_l = length(col)
        eps = 1e-12
        maximumsteps = 20000
        ccall((:greensfunctions,"./RSCG.so"),Nothing,
        (Ref{Int64},#i
        Ref{Int64},#j
        Ref{ComplexF64}, #σ
        Ref{Int64},#M
        Ref{Int64},#n
        Ref{ComplexF64},#val
        Ref{Int64},#row
        Ref{Int64},#col
        Ref{Int64},#val_l
        Ref{Int64},#col_l
        Ref{Int64},#row_l
        Ref{Float64},#eps
        Ref{Int64},#maximumsteps
        Ref{ComplexF64}),#Theta
        i,j,σ,M,n,val,row,col,val_l,row_l,col_l,eps,maximumsteps,Theta)
        return Theta
    end

    function greensfunctions_fortran(vec_left::Array{<:Integer,1},j,A,σ;eps=1e-12)
        nl = length(vec_left)
        n = size(A,1)
        M = length(σ)
        Theta = zeros(ComplexF64,M,nl)
        col = A.colptr
        val = A.nzval
        row = A.rowval
        val_l = length(val)
        row_l = length(row)
        col_l = length(col)
        #eps = 1e-12
        maximumsteps = 20000
        ccall((:greensfunctions_vec,"./RSCG.so"),Nothing,
        (Ref{Int64},#vec_left
        Ref{Int64},#nl
        Ref{Int64},#j
        Ref{ComplexF64}, #σ
        Ref{Int64},#M
        Ref{Int64},#n
        Ref{ComplexF64},#val
        Ref{Int64},#row
        Ref{Int64},#col
        Ref{Int64},#val_l
        Ref{Int64},#col_l
        Ref{Int64},#row_l
        Ref{Float64},#eps
        Ref{Int64},#maximumsteps
        Ref{ComplexF64}),#Theta
        vec_left,nl,j,σ,M,n,val,row,col,val_l,row_l,col_l,eps,maximumsteps,Theta)
        return Theta
    end

end
