CC Codice di Massimiliano D'Elia

CC####################################################################
      program ising
CC####################################################################
c programma per la simulazione del modello di ising bidimensionale
c con possibilita` di inserire un campo magnetico esterno
C--------------------------------------------------------------------
      parameter (nlatt = 10, nvol = nlatt**2)
      real field
      integer*8 seed,seed_old
      common/lattice/field(nlatt,nlatt) !! matrice con le variabili di spin
                                        !! che rappresenta lo stato del sistema
      common/various/beta,extfield      !! parametri della simulazione

CC    per far girare il codice dopo averlo compilato, e` necessario che 
CC    esista un file "input" contenente i 5 numeri (3 interi e 2 reali) letti
CC    poche righe piu` sotto, ed un file "randomseed" contenente lo stato
CC    precedente del generatore random oppure il nuovo seme 
CC    (un numero intero qualsiasi)

CC    inoltre se si mette come "iflag" di partenza un numero che non sia
CC    0 o 1, prova a leggere la configurazione di spin dal file "lattice"
CC    che quindi deve gia` esistere


      call ranstart                 !! initialize random number generator
                                    !! per chiamare un numero random usare
                                    !! la funzione ran2(), vedere gli esempi
                                    !! riportati piu` avanti per capire come
                                    !! si usa

cc apertura file da dove leggere i parametri della simulazione
      open(1,file='input',status='old')

cc apertura file sul quale scrivere le misure della magnetizzazione
      open(2,file='data.dat',status='unknown')

CC======================================
CC LETTURA PARAMETRI DELLA SIMULAZIONE
CC====================================== 
      read(1,*) iflag          !!partenza caldo(1)/freddo(0)/precedente(altro) 
      read(1,*) measures       !!numero di misure
      read(1,*) i_decorrel     !!updating fra una misura e l'altra
      read(1,*) extfield       !!valore del campo esterno
      read(1,*) beta           !!valore di 1/(kT) = beta
CC======================================

CC======================================
CC OPERAZIONI PRELIMINARI
CC======================================
      CALL geometry()                     !!inizializza condizioni al bordo
      CALL initialize_lattice(iflag,seed) !!inizializza configurazione iniziale
CC======================================

      do iter = 1,measures

CC   AGGIORNAMENTO CONFIGURAZIONE: i_decorrel spazzate di tutto il reticolo
         do idec = 1,i_decorrel
            call update_metropolis()
         enddo

CC   MISURA DELLE OSSERVABILI FISICHE
         call magnetization(xmagn)
         call energy(xene)

CC   SCRIVO LE MISURE SU FILE PER POI EFFETTUARE L'ANALISI
         write(2,*) iter,xmagn,xene

      enddo !! measures
CC=========TERMINE SIMULAZIONE MONTE-CARLO===========

CC SALVO CONFIGURAZIONE E STATO GEN. RANDOM PER 
CC POTER EVENTUALMENTE RIPARTIRE
      open(3,file='lattice',status='unknown')
      write(3,*) field
      close(3)      
      call ranfinish

CC==============================================
      stop
      end
CC===========F   I   N   E======================
CC===M  A   I   N      P  R  O  G  R  A  M======   





CC################################################################
CC      INIZIO SUBROUTINES
CC################################################################


c*****************************************************************
      subroutine geometry()    
c*****************************************************************
c per ogni coordinata definisco il passo in avanti o indietro
c con le opportune condizioni al bordo
c=================================================================
      parameter (nlatt = 10,nvol = nlatt**2)
      common/move/npp(nlatt),nmm(nlatt) 
      
      !! le funzioni npp ed nmm sono costruite come dei vettori
      !! di interi, in modo da non essere ricalcolate ogni volta
      !! e rendere tutto piu` efficiente, prendono in input una 
      !! coordinata e restituiscono la coordinata in avanti o
      !! indietro, tenendo conto delle opportune condizioni

      do i = 1,nlatt
         npp(i) = i + 1
         nmm(i) = i - 1
      enddo
      npp(nlatt) = 1             !! RIAGGIUSTO IL CALCOLO AI BORDI PER TENERE
      nmm(1) = nlatt             !! CONTO DELLE CONDIZIONI AL BORDO PERIODICHE

      return
      end
c=================================================================

c*****************************************************************
      subroutine initialize_lattice(iflag,seed)
c ASSEGNO LA CONFIGURAZIONE DI PARTENZA DELLA CATENA DI MARKOV
c=================================================================
      parameter (nlatt = 10,nvol = nlatt**2)
      common/lattice/field(nlatt,nlatt)
      integer*8 seed

CC  PARTENZA A FREDDO (tutti gli spin a 1 come se fosse T = 0)
      if (iflag.eq.0) then
         do i = 1,nlatt                  !! loop su tutti i siti
            do j = 1,nlatt               !! del reticolo
               field(i,j) = 1.0
            enddo
         enddo
CC  ... A CALDO ... (spin random, come se fosse T = infinito) 
      elseif (iflag.eq.1) then
         do i = 1,nlatt                  !! loop su tutti i siti
            do j = 1,nlatt               !! del reticolo
               x = ran2()               !! ran2() random fra 0 e 1
               field(i,j) = 1.0
               if (x.lt.0.5) field(i,j) = -1.0
            enddo
         enddo
CC  ... O DA DOVE ERO RIMASTO L'ULTIMA VOLTA
      else
         open(9,file='lattice',status='old')
         read(9,*) field
         read(9,*) seed  
         close(9)
      endif

      return
      end
c=================================================================

c*****************************************************************
      subroutine magnetization(xmagn)
c calcolo magnetizzazione media del reticolo
c*****************************************************************
      parameter (nlatt = 10,nvol = nlatt**2)
      common/lattice/field(nlatt,nlatt)

      xmagn = 0.0                       !! inizializzo xmagn a zero
      do i = 1,nlatt                    !! faccio il loop su tutto il reticolo 
         do j = 1,nlatt                 !! e sommo tutti i valori del campo
            xmagn = xmagn + field(i,j)  
         enddo                          
      enddo                             
      xmagn = xmagn/float(nvol)         !! normalizzo dividendo per il volume

      return
      end
c=================================================================

c*****************************************************************
      subroutine energy(xene)
C energia media (= 0 per configurazione ordinata e campo esterno 0)
c================================================================= 
      parameter (nlatt = 10,nvol = nlatt**2)
      common/lattice/field(nlatt,nlatt)
      common/move/npp(nlatt),nmm(nlatt)
      common/various/beta,extfield                 

      xene = 0.0                             !! inizializzo variabile
      do i = 1,nlatt                         !! inizio il loop sul reticolo 
         do j = 1,nlatt
            ip = npp(i)                      !! calcolo coordinate prime vicine
            im = nmm(i)                      !! del sito 
            jp = npp(j)
            jm = nmm(j)

            force = field(i,jp) + field(i,jm) +     !! somma dei 4 primi vicini
     $              field(ip,j) + field(im,j)
            xene = xene -  0.5*force*field(i,j) !! 1/2 per conteggio giusto
            xene = xene - extfield*field(i,j)       !! contributo campo esterno
         enddo
      enddo                                         
      xene = xene/float(nvol)            !! normalizzo -> densita` di energia

      return
      end
c=================================================================

c*****************************************************************
      subroutine update_metropolis()
cc faccio aggiornamenti locali delle variabili di spin con metropolis
cc la variabile di spin di prova e` sempre quella opposta a quella attuale
c*****************************************************************
      parameter (nlatt = 10,nvol = nlatt**2)
      common/lattice/field(nlatt,nlatt)
      common/move/npp(nlatt),nmm(nlatt)
      common/various/beta,extfield                  

      do ivol = 1,nlatt*nlatt            !! loop su tutti i siti

            i = int(ran2()*nlatt + 1.0)  !! scelgo a caso un sito del reticolo
            j = int(ran2()*nlatt + 1.0)  !! 

            ip = npp(i)         !! calcolo le coordinate
            im = nmm(i)         !! dei quattro primi vicini
            jp = npp(j)         !! del sito che ho scelto
            jm = nmm(j)
            
            force = field(i,jp) + field(i,jm) + !! calcolo la somma dei primi
     $              field(ip,j) + field(im,j)   !! quattro vicini, aggiungo il
            force = beta*(force + extfield)     !! campo esterno e divido per  
                                                !! (kT): ottengo quello che 
                                                !! moltiplicato per lo spin in
                                                !! (i,j) mi da la parte di 
                                                !! energia che dipende solo  
                                                !! dallo spin in (i,j), da cui
                                                !! deriva il rapporto di prob.

            phi =  field(i,j)           !! phi = valore attuale dello spin. 
                                        !! Il tentativo e` sempre invertire lo 
                                        !! lo spin. Calcolo il rapporto p_rat
            p_rat = exp(-2.0*phi*force) !! di prob. fra il caso invertito e non

            x = ran2()                       !! METRO-TEST! x = random (0,1)
                                              !! x < p_rat verifica anche caso 
            if (x.lt.p_rat) field(i,j) = -phi !! p_rat > 1 -> se si accetto


      enddo                                     !!  sui siti del reticolo

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
