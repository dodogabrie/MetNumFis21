      program harmonic_oscillator
      
      parameter (nlatt = 10)
      real lattice
      integer seed
      common/lattice/field(nlatt)
      common/various/eta,d_metro,seed


      open(1,file='input',status='old')
      open(2,file='meas_out',status='unknown')

CC====================================
CC LETTURA PARAMETRI DELLA SIMULAZIONE
CC==================================== 
      read(1,*) iflag          !!partenza caldo/freddo/precedente 
      read(1,*) measures       !!numero di misure
      read(1,*) i_decorrel     !!updating fra una misura e l'altra
      read(1,*) i_term         !!passi di termalizzazione
      read(1,*) d_metro        !!parametro del metropolis
      read(1,*) eta            !!valore del parametro eta = omega * a
CC=====================================
c     d_metro = 2.0*sqrt(eta)

CC=====================================
CC OPERAZIONI PRELIMINARI
CC=====================================
      call ranstart                     !! initialize random number generator
      CALL initialize_lattice(iflag)    !! inizializza configurazione iniziale
      CALL geometry()                   !! inizializza condizioni al bordo
CC=====================================


CC=============================
CC TERMALIZZAZIONE
CC=============================
      do it = 1,i_term
         call update_metropolis()
      enddo
CC=============================


CC=============================================================
CC SESSIONE ALL'EQUILIBRIO CON MISURE
CC=============================================================
      do iter = 1,measures

CC   AGGIORNAMENTO CONFIGURAZIONE
         do idec = 1,i_decorrel
            call update_metropolis()
         enddo

         call measure()

      enddo !! measures
CC=========TERMINE SIMULAZIONE MONTE-CARLO===========



CC==============================================
CC PRENDO L'ULTIMO STATO DEL GENERATORE RANDOM
CC==============================================
      call ranfinish

CC==============================================
CC SALVO CONFIGURAZIONE E STATO GEN. RANDOM PER POTER RIPARTIRE
CC==============================================
      open(3,file='lattice',status='unknown')
      write(3,*) field
      close(3)      

CC==============================================
      stop
      end
CC===========F   I   N   E======================




CC      INIZIO DEFINIZIONE SUBROUTINES

c*****************************************************************
      subroutine geometry()    
c*****************************************************************
c DEFINISCO LA COORDINATA + e - 
c=================================================================
      parameter (nlatt = 10)
      common/move/npp(nlatt),nmm(nlatt)
      
      do i = 1,nlatt
         npp(i) = i + 1
         nmm(i) = i - 1
      enddo
      npp(nlatt) = 1             !!        CONDIZIONI AL BORDO
      nmm(1) = nlatt             !!             PERIODICHE

      return
      end
c=================================================================


c*****************************************************************
      subroutine initialize_lattice(iflag)
c*****************************************************************
c ASSEGNO LA CONFIGURAZIONE DI PARTENZA DELLA CATENA DI MARKOV
c=================================================================
      parameter (nlatt = 10)

      common/lattice/field(nlatt)
      common/various/eta,d_metro,seed

CC  PARTENZA A FREDDO ...
      if (iflag.eq.0) then
         do i = 1,nlatt                  !! loop su tutti i siti
               field(i) = 0.0
         enddo
CC  ... A CALDO ...
      elseif (iflag.eq.1) then
         do i = 1,nlatt                  !! loop su tutti i siti
               x = 1.0 - 2.*ran2()       !! frand() random fra -1 e 1
                  field(i) = x
         enddo
CC  ... O DA DOVE ERO RIMASTO L'ULTIMA VOLTA
      else
         open(9,file='lattice',status='old')
         read(9,*) field
c         read(9,*) seed  
         close(9)
      endif

      return
      end
c=================================================================


c*****************************************************************
      subroutine update_metropolis()
c*****************************************************************
 
      parameter (nlatt = 10)
      common/lattice/field(nlatt)
      common/move/npp(nlatt),nmm(nlatt)
      common/various/eta,d_metro,seed                  

      c1 = 1./eta
      c2 = (1./eta + eta/2.)

      do i = 1,nlatt            !! loop su tutti i siti, qui il sito 
                                !! non e` scelto a caso ma faccio una spazzata 
                                !! iterativa su tutti i siti, si puo` dimostrare
                                !! che va bene lo stesso per il bilancio, ma meno banale da provare 
         
         ip = npp(i)            !! calcolo le coordinate
         im = nmm(i)            !! dei due primi vicini
         
         force = field(ip) + field(im) !! costruisco la forza
         
         phi =  field(i)        !! phi = valore attuale del campo. 
         phi_prova = phi + 2.*d_metro*(0.5-ran2())                  

         p_rat = c1 * phi_prova * force - c2 * phi_prova**2
         p_rat = p_rat - c1 * phi * force + c2 * phi**2
         
         
         x = log(ran2())                      !! METRO-TEST! x = random (0,1)
                                              !! x < p_rat verifica anche caso 
         if (x.lt.p_rat) field(i) = phi_prova !! p_rat > 1 -> se si accetto
         
      enddo                     !!  chiudo il loop

      return
      end
c=================================================================


c*****************************************************************
      subroutine measure()
c*****************************************************************

      parameter (nlatt = 10)
      common/lattice/field(nlatt)
      common/move/npp(nlatt),nmm(nlatt)

      obs1 = 0.0
      obs2 = 0.0
      do i = 1,nlatt        
        obs1 = obs1 + field(i)**2
        obs2 = obs2 + (field(i)-field(npp(i)))**2
      enddo                             

      obs1 = obs1/float(nlatt) !! media sul singolo path di y^2 
      obs2 = obs2/float(nlatt) !! media sul singolo path di Delta y^2 
c      write(2,*) obs1,obs2

      return
      end
c=================================================================


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


