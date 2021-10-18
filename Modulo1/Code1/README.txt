per chi avesse scritto il proprio codice di Ising in 2D e volesse
verificare se e` giusto, riporto il valore dell'energia media 
ottenuto con il mio codice usando questi parametri:

- reticolo 10 x 10 con condizioni periodiche

- beta = 0.3

- h (campo esterno) = 0

- risultato dopo 100K misure (una fatta ogni 100 chiamate della
  funzione "update_metropolis", quindi dopo 100 x 10 x 10 = 10000
  chiamate del metropolis di singolo sito)

  densita` di energia media = <E> / V = -0.7067(6) (errore 6 sull'ultima cifra)

  il volume V in questo caso vale ovviamente 10 x 10 = 100

  invece la magnetizzazione media dovrebbe venire zero entro gli errori

  il tempo totale di esecuzione sul mio laptop e` di circa 70 secondi

