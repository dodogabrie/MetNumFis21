import numpy as np
from oscillatore import simulator


def metropolis_harmonic_osc():
    nlat = 10 
    eta = 0.1 
    d_metro = 0.5 
    measures = int(1e7)
    i_decorrel = 10 
    i_term = int(1e6)
    seed = 0
    iflag = 0
    simulator(seed, nlat, iflag, 
              measures, i_decorrel, i_term, d_metro,
              eta, save_lattice = 0)
    return


if __name__ == '__main__':
    pass
#    metropolis_harmonic_osc()

