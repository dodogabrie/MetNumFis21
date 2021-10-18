CC Codice di Massimo D'Elia

cc============================================
      subroutine metrogauss()
cc============================================
cc the aim is to generate gaussian number with
cc given average and given variance using a
cc metropolis algorithm as a toy model
cc============================================

      implicit real (a-h,o-z)
      implicit integer (i-n)
      call ranstart                 !! initialize random number generator

      q = start               !!inizializzo lo stato del mio sistema 

      open(1,file='inputGauss',status='old')
CC======================================
CC LETTURA PARAMETRI DELLA SIMULAZIONE
CC====================================== 
      read(1,*) nstat      
      read(1,*) start     
      read(1,*) aver     
      read(1,*) sigma2  
      read(1,*) delta  
      close(1)

      open(2,file='data.dat',status='unknown')
      
      do i = 1,nstat
         x = ran2()                        !! due numeri random fra 0 e 1  
         y = ran2()                        !! che mi serviranno dopo


         q_try = q + delta*(1.0 - 2.0*x)!! mi muovo in un intervallo    
                                            !! +- delta intorno al vecchio 
                                            !! valore  
         
         z = exp(((q-aver)**2 - (q_try-aver)**2)/2.0/sigma2)  !! rapporto fra le probabilita`

         if(y.lt.z) then                    !! accept reject 
            q = q_try
            acc = 1.0                       !! accettanza = 1 o 0
         else
            acc = 0.0                       !! se non accetto tengo q vecchio
         endif

         write(2,*) i,q,acc                  

      enddo

      call ranfinish
      end

c============================================================================
c  RANDOM NUMBER GENERATOR: standard ran2 from numerical recipes
c============================================================================
      function ran2()
      implicit real*4 (a-h,o-z)
      implicit integer*4 (i-n)
      integer idum,im1,im2,imm1,ia1,ia2,iq1,iq2,ir1,ir2,ntab,ndiv
      real ran2,am,eps,rnmx
      parameter(im1=2147483563,im2=2147483399,am=1./im1,imm1=im1-1,
     &          ia1=40014,ia2=40692,iq1=53668,iq2=52774,ir1=12211,
     &          ir2=3791,ntab=32,ndiv=1+imm1/ntab,eps=1.2e-7,
     &          rnmx=1.-eps)
      integer idum2,j,k,iv,iy
      common /dasav/ idum,idum2,iv(ntab),iy
c      save iv,iy,idum2
c      data idum2/123456789/, iv/NTAB*0/, iy/0/

      if(idum.le.0) then
         idum=max0(-idum,1)
         idum2=idum
         do j=ntab+8,1,-1
            k=idum/iq1
            idum=ia1*(idum-k*iq1)-k*ir1
            if(idum.lt.0) idum=idum+im1
            if(j.le.ntab) iv(j)=idum
         enddo
         iy=iv(1)
      endif
      k=idum/iq1
      idum=ia1*(idum-k*iq1)-k*ir1
      if(idum.lt.0) idum=idum+im1
      k=idum2/iq2
      idum2=ia2*(idum2-k*iq2)-k*ir2
      if(idum2.lt.0) idum2=idum2+im2
      j=1+iy/ndiv
      iy=iv(j)-idum2
      iv(j)=idum
      if(iy.lt.1) iy=iy+imm1
      ran2=min(am*iy,rnmx)

      return
      end

c=============================================================================
      subroutine ranstart
      implicit real*4 (a-h,o-z)
      implicit integer*4 (i-n)
      common /dasav/ idum,idum2,iv(32),iy

      open(unit=23, file='randomseed', status='unknown')
      read(23,*) idum
      read(23,*,end=117) idum2
      do i=1,32
         read(23,*) iv(i)
      enddo
      read(23,*) iy
      close(23)
      goto 118                          !!takes account of the first start
 117  if(idum.ge.0) idum = -idum -1     !!
      close(23)
 118  continue                          !!

      return
      end

c=============================================================================
      subroutine ranfinish
      implicit real*4 (a-h,o-z)
      implicit integer*4 (i-n)
      common /dasav/ idum,idum2,iv(32),iy

      open(unit=23, file='randomseed', status='unknown')
      write(23,*) idum
      write(23,*) idum2
      do i=1,32
         write(23,*) iv(i)
      enddo
      write(23,*) iy
      close(23)

      return
      end
c=============================================================================



