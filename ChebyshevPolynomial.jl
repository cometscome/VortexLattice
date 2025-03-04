module Chebyshev

    export calc_meanfields
    using SparseArrays
    using LinearAlgebra
    using Distributed

    function calc_meanfields(nc,A,Nx,Ny,aa,bb,ωc)
        cc = spzeros(typeof(A[1,Nx*Ny]),Nx*Ny,Nx*Ny)
        vec_c = pmap(ii -> calc_meanfield_i(ii,nc,A,aa,bb,ωc,Nx,Ny),1:Nx*Ny)
        for ii=1:Nx*Ny
            cc[ii,ii] = vec_c[ii]
        #calc_meanfield_i(ii,nc,A,aa,bb,ωc,Nx,Ny)
        end
        #=
       for ix=1:Nx
            for iy=1:Ny
                ii = (iy-1)*Nx+ix
                jj = ii + Nx*Ny
                vec_ai = calc_polynomials(nc,ii,jj,A)
                cc[ii,ii] = calc_meanfield(vec_ai,aa,bb,ωc,nc)
            end
        
end
    =#
        return cc
    end

    function calc_meanfield_i(ii,nc,A,aa,bb,ωc,Nx,Ny)
        jj = ii + Nx*Ny
        #vec_ai = calc_polynomials_easytoread(nc,ii,jj,A)
        vec_ai = calc_polynomials(nc,ii,jj,A)
        return calc_meanfield(vec_ai,aa,bb,ωc,nc)
    end

    function calc_meanfields(A,Nx,Ny,ωc)
       
        cc = spzeros(typeof(A[1,Nx*Ny]),Nx*Ny,Nx*Ny)
        A = Matrix(A)
        w,v = eigen(A)        
        
    
        for ix=1:Nx
            for iy=1:Ny
                ii = (iy-1)*Nx+ix
                jj = ii + Nx*Ny
            
                cc[ii,ii] = 0.0
                for i=1:Nx*Ny*2
                    if w[i] <= 0.0
                        if abs(w[i]) <= ωc
                            #cc[ii,ii]+=conj(v[ii,i])*v[jj,i]
                            cc[ii,ii]+=conj(v[jj,i])*v[ii,i]
                        end
                    end
                end
            end
        end
        A = sparse(A)
        return cc
    end

    function calc_meanfield(vec_ai,aa,bb,ωc,nc)
        ba = acos(-bb/aa)
        ωb = acos(-(ωc+bb)/aa)
        density = vec_ai[0+1]*(ωb-ba)/2
        for nn=1:nc-1
            density += vec_ai[nn+1]*(sin(nn*ωb)-sin(nn*ba))/nn
        end
        density = density*2/π
    end

    function calc_polynomials(nc,left_i,right_j,A)
        Ln = size(A,1) 
        
        typeA = eltype(A)

        vec_ai = zeros(typeA,nc)
        vec_js = Array{Array{typeA,1}}(undef,2)
        ptr = Int8[1,2] #This is a pointer array
        vec_js[ptr[1]] = zeros(typeA,Ln)
        vec_js[ptr[2]] = zeros(typeA,Ln)

        vec_js[ptr[1]][right_j] = 1.0

        nn = 0
        vec_ai[nn+1] = vec_js[ptr[1]][left_i]
        nn = 1
        mul!(vec_js[ptr[2]], A, vec_js[ptr[1]])
        vec_ai[nn+1] = vec_js[ptr[2]][left_i]

        @inbounds for nn=2:nc-1
            #mul!(C, A, B, α, β) -> C   A B α + C β
            mul!(vec_js[ptr[1]],A,vec_js[ptr[2]],2,-1)
            vec_ai[nn+1] = vec_js[ptr[1]][left_i]
            ptr[1],ptr[2] = ptr[2],ptr[1]
        end

        return vec_ai
    end

    function calc_polynomials_easytoread(nc,left_i,right_j,A)
        Ln = size(A,1) 
        typeA = typeof(A[1,1])
        vec_jnmm = zeros(typeA,Ln)
        vec_jnm = zeros(typeA,Ln)
        vec_jn = zeros(typeA,Ln)
        vec_jn[right_j] = 1.0
        vec_ai = zeros(typeA,nc)
        @inbounds for nn=0:nc-1
            if nn == 0
                vec_jn[right_j] = 1.0
            elseif nn == 1
                mul!(vec_jn, A, vec_jnm)
                #A_mul_B!(vec_jn,A,vec_jnm)
            else
                mul!(vec_jn, A, vec_jnm)
                #A_mul_B!(vec_jn,A,vec_jnm)
#                vec_jn *= 2
                for i=1:Ln
                    vec_jn[i] = vec_jn[i]*2 -vec_jnmm[i]
                end                                               
            end
             
            vec_ai[nn+1] = vec_jn[left_i]
            #println("vec_a, ",vec_ai[nn+1])
            for i=1:Ln
                vec_jnmm[i] = vec_jnm[i]
                vec_jnm[i] = vec_jn[i]
            end


        end
        return vec_ai
    end
end