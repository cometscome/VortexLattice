module rscg
contains
  subroutine greensfunctions(i,jj,vec_sigma,M,n,val,row,col,&
    val_l,row_l,col_l,eps,maximumsteps,Theta) bind(C, name="greensfunctions")
    implicit none
    integer(8),intent(in)::M,n
    complex(8),intent(in)::vec_sigma(1:M)
    integer(8),intent(in)::val_l,col_l,row_l
    complex(8),intent(in)::val(1:val_l)
    integer(8),intent(in)::col(1:col_l),row(1:row_l)
    integer(8),intent(in)::i,jj
    complex(8),intent(out)::Theta(1:M)
    real(8),intent(in),optional::eps
    integer(8),intent(in),optional::maximumsteps
    complex(8),allocatable::b(:),x(:),r(:),p(:),Ap(:)
    complex(8)::alpham,betam,Sigma
    complex(8),allocatable::Pi(:)
    complex(8)::rhok(1:M),rhokm(1:M),rhokp(1:M)
    integer::k,j
    real(8)::rnorm,hi
    complex(8)::alpha,beta,alphakj,betakj

    allocate(b(1:N))
    b = 0d0
    b(jj) = 1d0
    allocate(x(1:N))
    x = 0d0
    allocate(r(1:N))
    r = b
    allocate(p(1:N))
    p = b
    alpham = 1d0
    betam = 1d0
    Sigma = b(i)
    Theta = 0d0
    allocate(Pi(1:M))
    Pi = Sigma
    rhok = 1d0
    rhokm = rhok
    rhokp = rhok
    allocate(Ap(1:N))
    Ap = 0d0
    do k=0,maximumsteps
      call mkl_zcsrgemv("T", n, val, col, row, -p, Ap)
      rnorm = dot_product(r,r)
      alpha = rnorm/dot_product(p,Ap)
      x = x + alpha*p
      r = r -alpha*Ap
      beta = dot_product(r,r)/rnorm
      p = r + beta*p
      Sigma = r(i)
      do j=1,M
        if(abs(rhok(j))> eps) then
          rhokp(j) = rhok(j)*rhokm(j)*alpham/(rhokm(j)*alpham* &
            (1d0+alpha*vec_sigma(j))+alpha*betam*(rhokm(j)-rhok(j)))
          alphakj = alpha*rhokp(j)/rhok(j)
          Theta(j) = Theta(j) + alphakj*Pi(j)
          betakj = ((rhokp(j)/rhok(j))**2)*beta
          Pi(j) = rhokp(j)*Sigma+betakj*Pi(j)
        endif
        rhokm(j) = rhok(j)
        rhok(j) = rhokp(j)
      enddo
      alpham = alpha
      betam = beta
      hi = rnorm*maxval(abs(rhok))
      if (hi < eps) then
        return
      endif

    enddo



    write(*,*) "not converged"
    return


  end

  subroutine greensfunctions_vec(vec_left,nl,jj,vec_sigma,M,n,val,row,col,val_l,&
    row_l,col_l,eps,maximumsteps,Theta) bind(C, name="greensfunctions_vec")
    implicit none
    integer(8),intent(in)::nl
    integer(8),intent(in)::vec_left(1:nl)
    integer(8),intent(in)::M,n
    complex(8),intent(in)::vec_sigma(1:M)
    integer(8),intent(in)::val_l,col_l,row_l
    complex(8),intent(in)::val(1:val_l)
    integer(8),intent(in)::col(1:col_l),row(1:row_l)
    integer(8),intent(in)::jj
    complex(8),intent(out)::Theta(1:M,1:nl)
    real(8),intent(in),optional::eps
    integer(8),intent(in),optional::maximumsteps
    complex(8),allocatable::b(:),x(:),r(:),p(:),Ap(:)
    complex(8)::alpham,betam,Sigma(1:nl)
    complex(8),allocatable::Pi(:,:)
    complex(8)::rhok(1:M),rhokm(1:M),rhokp(1:M)
    integer::k,j,mm
    real(8)::rnorm,hi
    complex(8)::alpha,beta,alphakj,betakj

    allocate(b(1:N))
    b = 0d0
    b(jj) = 1d0
    allocate(x(1:N))
    x = 0d0
    allocate(r(1:N))
    r = b
    allocate(p(1:N))
    p = b
    alpham = 1d0
    betam = 1d0
    do mm=1,nl
      Sigma(mm) = b(vec_left(mm))
    enddo
    Theta = 0d0
    allocate(Pi(1:M,1:nl))
    do mm=1,nl
      Pi(:,mm) = Sigma(mm)
    enddo
    rhok = 1d0
    rhokm = rhok
    rhokp = rhok
    allocate(Ap(1:N))
    Ap = 0d0
    do k=0,maximumsteps
      call mkl_zcsrgemv("T", n, val, col, row, -p, Ap)
      rnorm = dot_product(r,r)
      alpha = rnorm/dot_product(p,Ap)
      x = x + alpha*p
      r = r -alpha*Ap
      beta = dot_product(r,r)/rnorm
      p = r + beta*p
      do mm=1,nl
        Sigma(mm) = r(vec_left(mm))
      enddo
      do j=1,M
        if(abs(rhok(j))> eps) then
          rhokp(j) = rhok(j)*rhokm(j)*alpham/(rhokm(j)*alpham* &
            (1d0+alpha*vec_sigma(j))+alpha*betam*(rhokm(j)-rhok(j)))
          alphakj = alpha*rhokp(j)/rhok(j)
          Theta(j,:) = Theta(j,:) + alphakj*Pi(j,:)
          betakj = ((rhokp(j)/rhok(j))**2)*beta
          Pi(j,:) = rhokp(j)*Sigma(:)+betakj*Pi(j,:)
        endif
        rhokm(j) = rhok(j)
        rhok(j) = rhokp(j)
      enddo
      alpham = alpha
      betam = beta
      hi = rnorm*maxval(abs(rhok))
      if (hi < eps) then
        return
      endif

    enddo



    write(*,*) "not converged"
    return


  end
end
