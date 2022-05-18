import numpy as np
from joblib import Parallel, delayed
from oscillatore import simulator
from oscillatore_f2connessa import simulator_f2


def metropolis_harmonic_osc(nlat, eta, measures, d_metro = None, 
                            i_decorrel = 10, i_term = None, seed = -1, 
                            iflag = 1, save_data = 1, save_lattice = 1, data_dir = "", file_name = None, input_list_k = []):
    if d_metro is None: d_metro = 2*np.sqrt(eta)
    if i_term  is None: i_term = measures

    if input_list_k is []:
        simulator(nlat, iflag, 
                  measures, i_decorrel, i_term, d_metro,
                  eta, save_data = save_data, save_lattice = save_lattice, seed = seed, data_dir = data_dir, file_name = file_name)
    else:
        simulator_f2(input_list_k, nlat, iflag,
                    measures, i_decorrel, i_term, d_metro,
                    eta, save_data = save_data, save_lattice = save_lattice, seed = seed, data_dir = data_dir, file_name = file_name)
    return

def simulation_varying_nlat(n_jobs, nlat_list, eta, measures, *args, **kwargs):
    # parallel version:
    Parallel(n_jobs=n_jobs)(delayed(metropolis_harmonic_osc)(nlat, eta, measures, *args, **kwargs) 
            for nlat in nlat_list)
    # It's the same as:
    # for nlat in nlat_list:
    #     metropolis_harmonic_osc(nlat, eta, measures, *args, **kwargs)

def simulation_varying_nlat_and_eta(n_jobs, nlat_list, eta_list, measures, *args, **kwargs):
    # parallel version:
    Parallel(n_jobs=n_jobs)(delayed(metropolis_harmonic_osc)(nlat, eta, measures, *args, **kwargs) 
            for nlat, eta in zip(nlat_list, eta_list))
    # It's the same as:
    # for nlat, eta in zip(nlat_list, eta_list):
    #     metropolis_harmonic_osc(nlat, eta, measures, *args, **kwargs)


# Vedere i cammini al variare di N (nlat) ad eta fisso
def MC_story_varying_N():
    # SCELTA DEL VALORE DI eta: 
    # - eta ~ 1 => errori sistematici.
    # - eta -> 0 => Nessun errore sistematico ma simulazioni lunghissime.
    # - eta = 1e-3 => Compromesso tra tempo di simulazione ed errori sistematici.
    eta = 1e-2
    # Lista di N in analisi.
    list_N = [10, 20, 30, 40, 50, 70, 100, 300, 500, 1000, 2000, 5000]
    measures = int(1e6)
    n_jobs = 6
    data_dir = "fix_eta_varying_N"
    simulation_varying_nlat(n_jobs, list_N, eta, measures, i_decorrel = 100, data_dir = data_dir)

# Teniamo T fisso e facciamo variare N, eta
def potential_term(N_eta):
    n_jobs = 5
    list_eta = np.array([0.6, 0.3, 0.2, 0.15, 0.12, 0.10, 0.05, 0.0375, 0.03])
    list_N = (N_eta/list_eta).astype(int)
    measures = int(1e6)
    simulation_varying_nlat_and_eta(n_jobs, list_N, list_eta, measures, i_term = int(1e6), i_decorrel = 500)

def stato_fondamentale():
    data_dir = "stato_fondamentale"
    measures = int(1e2)
    i_decorrel = int(1e4)
    i_term = 0
    eta = 0.01
    nlat = 10000
    N_lattice = 20
    list_filename = [f'lattice{i}' for i in range(N_lattice)]
    for file_name in list_filename:
        print(file_name, 'of', N_lattice)
        metropolis_harmonic_osc(nlat, eta, measures,
                                i_decorrel = i_decorrel, i_term = i_term, save_data = 0, 
                                save_lattice = 1, data_dir = data_dir, file_name = file_name)

def gap_energy():
    data_dir = "gap_energy"
    i_term = int(1e6)
    measures = int(1e5)
    i_decorrel = 10
    eta = 0.8
    nlat = 5000
    wt_max = 5  # k eta
    K_max = np.ceil(wt_max/eta)
    if K_max == 0: raise ValueError
    K_min = 0
    input_list_k = np.arange(K_min, K_max)
    metropolis_harmonic_osc(nlat, eta, measures, i_term=i_term, i_decorrel = i_decorrel,
                            data_dir = data_dir, file_name=f'{eta}_{nlat}', input_list_k = input_list_k)

def contiuum_E1_E0():
    data_dir = "lim_continuum_E1_E0"
    tau_ext = 20 # temporal extension (i.e. beta omega)
    eta_list = [0.2, 0.4, 0.6, 0.8]
    eta_list = np.array(eta_list)
    N_list = (tau_ext / eta_list).astype(int)
    print(N_list)
    i_term = int(1e3)
    measures = int(1e5)
    i_decorrel = 1
    wt_max = 5  # k eta
    for eta, nlat in zip(eta_list, N_list):
        K_max = np.ceil(wt_max/eta)
        if K_max == 0: raise ValueError
        K_min = 0
        input_list_k = np.arange(K_min, K_max)
        metropolis_harmonic_osc(nlat, eta, measures, i_term=i_term, i_decorrel = i_decorrel, 
                                data_dir = data_dir, file_name=f'{eta}_{nlat}', input_list_k = input_list_k)

    return

# altro ...
#    n_jobs = 1
#    nlat_list = [20, 80, 50, 100, 200, 300, 400, 500, 1000, 5000]
#    eta = 1e-2
#    measures = int(1e6)
#    simulation_varying_nlat(n_jobs, nlat_list, eta, measures, i_term = int(1e6), i_decorrel = 10)

if __name__ == '__main__':
#    MC_story_varying_N()
#    potential_term(3)
#    gap_energy()
    contiuum_E1_E0()

