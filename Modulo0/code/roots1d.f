C Compute roots using bisection and newton-raphson
        implicit real*8 (a-h,q-z)      !C tutte le variabili tra a ed h, q e z                                      sono ad 8 bite 
        f0(x) = x*x*x/5. + x/5. + 0.1
        f1(x) = 3.*x*x/5. + 1./5

C Bisection
        x0 = -1.8
        x1 = 1.8
        fs = f0(x0)
        do i = 1,20
           xm = 0.5*(x1+x0)  !C middle point
           fm = f0(xm)       !C evaluete function in middle point
           if (fm*fs.gt.0.) then   !C if fm and fs=f(x0) have the same sign
              x0 = xm
             else
              x1 = xm
           endif
           print *,i,xm,abs(x1-x0),f0(xm)
        enddo
        print *,0.5*(x1+x0)

C Newton

        x0 = 1.9
        do i = 1,20
           x0 = x0 - f0(x0)/f1(x0)
           print *,i,x0,f0(x0)
        enddo
        stop
        end
