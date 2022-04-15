import numpy as np
from oscillatore import simulator


def metropolis_harmonic_osc(nlat, eta, measures, d_metro = None, 
                            i_decorrel = 10, i_term = None, seed = 0, 
                            iflag = 0, save_data = 1, save_lattice = 1):
    if d_metro is None: d_metro = 2*np.sqrt(eta)
    if i_term  is None: i_term = measures

    simulator(seed, nlat, iflag, 
              measures, i_decorrel, i_term, d_metro,
              eta, save_data = save_data, save_lattice = save_lattice)
    return

def simulation_varying_nlat(nlat_list, eta, measures, *args, **kwargs):
    for nlat in nlat_list:
        print('nlat = ', nlat)
        metropolis_harmonic_osc(nlat, eta, measures, *args, **kwargs)


if __name__ == '__main__':

    nlat_list = [200, 300, 400, 500, 1000, 5000]
    eta = 1e-3  # eta
    measures = int(1e6)
    simulation_varying_nlat(nlat_list, eta, measures)

