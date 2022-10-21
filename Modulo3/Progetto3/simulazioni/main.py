import numpy as np
from joblib import Parallel, delayed
from oscillatore import simulator
from oscillatore_f2connessa import simulator_f2
from oscillatore_stato_fondamentale import simulator as simulator_fond


def metropolis_harmonic_osc(nlat, eta, measures, d_metro = None, 
                            i_decorrel = 10, i_term = None, seed = -1, 
                            i_flag = 0, save_data = 1, save_lattice = 1, data_dir = "", file_name = None, input_list_k = [], verbose = 10):
    if d_metro is None: d_metro = 2 * np.sqrt(eta)
    if i_term  is None: i_term = measures

    if len(input_list_k) == 0:
        print('lancio simulatore per y2, dy2')
        simulator(nlat, i_flag, 
                  measures, i_decorrel, i_term, d_metro,
                  eta, save_data = save_data, save_lattice = save_lattice, seed = seed, data_dir = data_dir, file_name = file_name, verbose = verbose)
    else:
        print('lancio simulatore per funzione di correlazione connessa')
        simulator_f2(input_list_k, nlat, i_flag,
                    measures, i_decorrel, i_term, d_metro,
                    eta, save_data = save_data, save_lattice = save_lattice, seed = seed, data_dir = data_dir, file_name = file_name, verbose = verbose)
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
    i_decorrel_try = kwargs.get('i_decorrel')

    if isinstance(i_decorrel_try, np.ndarray):
        kwargs.pop('i_decorrel')
        print(kwargs)
        Parallel(n_jobs=n_jobs)(delayed(metropolis_harmonic_osc)(nlat, eta, measures, *args, i_decorrel = i_decorrel, **kwargs) 
                for nlat, eta, i_decorrel in zip(nlat_list, eta_list, i_decorrel_try))
    else:
        Parallel(n_jobs=n_jobs)(delayed(metropolis_harmonic_osc)(nlat, eta, measures, *args, **kwargs) 
                for nlat, eta in zip(nlat_list, eta_list))
    # It's the same as:
    # for nlat, eta in zip(nlat_list, eta_list):
    #     metropolis_harmonic_osc(nlat, eta, measures, *args, **kwargs)


def plot_cammini():
    data_dir = "plot_cammini/"
    #etas = [1e-1, 1e-2, 1e-3]
    eta = 1e-3
    measures = 1#int(5e5)
    i_decorrel = 1#int(1e6)
    i_term = int(1e6)
    i_flag = 0
    nlats = [ 100,]# 200, 1000]
    #nlats = [300]*len(etas)# 1000, 5000, 7000]
    n_jobs = 1
#    simulation_varying_nlat_and_eta(n_jobs, nlats, etas, measures, i_decorrel = i_decorrel, 
#                            i_flag = i_flag, i_term = i_term, data_dir = data_dir, 
#                            save_lattice = 0, save_data = 1)
    simulation_varying_nlat(n_jobs, nlats, eta, measures, i_decorrel = i_decorrel, 
                            i_flag = i_flag, i_term = i_term, data_dir = data_dir, save_lattice = 1, save_data = 0)
    return

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
def potential_term():
    n_jobs = 8
    N_per_eta = 3
    list_N = np.array([8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 23, 25, 30, 50, 70])
    list_eta = N_per_eta/list_N
    i_term = int(1e6)
    measures = int(1e6)
    i_decorrel_list = 100*(1/np.sqrt(list_eta)).astype(int)
    data_dir = "potential_term"
    print(i_decorrel_list)
    simulation_varying_nlat_and_eta(n_jobs, list_N, list_eta, measures, save_lattice = 0, i_flag = 0,
            i_term = i_term, i_decorrel = i_decorrel_list, data_dir = data_dir)



def identical_simulation_parallel(n_jobs, filenames, nlat, eta, measures, *args, **kwargs):
    # parallel version:
    try: kwargs.pop('file_name')
    except KeyError: pass
    Parallel(n_jobs=n_jobs)(delayed(metropolis_harmonic_osc)(nlat, eta, measures, file_name = name, *args, **kwargs) 
            for name in filenames)
 

def stato_fondamentale(): # Forma dello stato fondamentale 
    data_dir = "stato_fondamentale"
    measures = 1
    i_decorrel = 1
    i_flag = 1
    i_term = int(1e6)
    eta = 0.1
    nlat = 500
    N_lattice = 100
    list_filename = [f'lattice{i}' for i in range(N_lattice)]
    n_jobs = 5
    identical_simulation_parallel(n_jobs, list_filename, nlat, eta, measures,
                                  i_decorrel = i_decorrel, i_term = i_term, save_data = 0, i_flag = i_flag,
                                  save_lattice = 1, data_dir = data_dir)

def stato_fondamentale_singola(): # Forma dello stato fondamentale 
    measures = 1000
    num_simulations = 50
    i_decorrel = 1000
    i_flag = 1
    i_term = int(1e6)
    eta = 0.1
    nlat = 12
    d_metro = 2 * np.sqrt(eta)
    folderdir = f'stato_fondamentale_singola/nlat{nlat}_eta{eta}/'
    njobs = 5
    list_filename = [f'lattice_singola{i}' for i in range(num_simulations)]
    def single_sim_fond(nlat, i_flag, 
                       measures, i_decorrel, i_term, d_metro,
                       eta, folderdir, filename, single_file = 0):
        simulator_fond(nlat, i_flag, 
                       measures, i_decorrel, i_term, d_metro,
                       eta, data_dir = folderdir, file_name = filename, single_file = single_file)
        print('Simulation '+ filename + ' done')
    Parallel(n_jobs=njobs)(delayed(single_sim_fond)(nlat, i_flag, 
                                                    measures, i_decorrel, i_term, d_metro,
                                                    eta, folderdir, filename, single_file = 1) 
                                                    for filename in list_filename)


def gap_energy(): # Misura di un singolo gap di energia ( singolo fit esponenziale)
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

def contiuum_E1_E0(): # limite al continuo per E1 e E0
    tau_ext = 60 # temporal extension (i.e. beta omega)
    data_dir = "lim_continuum_E1_E0" + f"/beta_omega_{tau_ext}" + '_test'
    #N_list = np.array([200, 100, 70, 60, 55, 50, 40, 37, 35, 33, 30, 29, 28, 27, 26, 25, 24, 23, ])
    N_list = np.array([50, 30, ])#25, 20, ])
    eta_list = tau_ext / N_list
    print('eta values: ', eta_list)
    print('N values: ', N_list)
    print('tau values: ', N_list * eta_list)
    i_term = int(1e5)
    measures = int(1e4)
#    i_decorrel_list = 100*(1/np.sqrt(eta_list)).astype(int) + 50
    i_decorrel_list = 20*np.ones(len(eta_list))
    verbose = 10
    wt_max = 6  # k eta
    def single_simulation(eta, nlat, i_decorrel):
        print('Simulation values: eta:', eta, 'nlat:', nlat)
        K_max = np.ceil(wt_max/eta)
        if K_max == 0: print('Warning: no K to simulate'); return
        K_min = 0
        input_list_k = np.arange(K_min, K_max)
        metropolis_harmonic_osc(nlat, eta, measures, i_term=i_term, i_decorrel = i_decorrel, i_flag=1,
                                data_dir = data_dir, file_name=f'{eta}_{nlat}', input_list_k = input_list_k, verbose = verbose)
    print('Start simulation')
    n_jobs = 4
    Parallel(n_jobs=n_jobs)(delayed(single_simulation)(eta, nlat, i_dec) for eta, nlat, i_dec in zip(eta_list, N_list, i_decorrel_list))


# altro ...
def U_varying_T():
    print('U_varying_T')
    n_jobs = 2
    #nlat_list = np.array([20, 30, 50, 80, 100, 200, 300, 400])
    #nlat_list = np.array([25, 40, 70, 150])
    nlat_list = np.array([800, 1500])
    eta = 1e-2
    i_term = int(1e6)
    measures = int(1e6)
    i_decorrel = 1500
    data_dir = "U_varying_T"
    simulation_varying_nlat(n_jobs, nlat_list, eta, measures, i_term = i_term, 
            i_decorrel = i_decorrel, data_dir = data_dir, save_lattice = 0)

if __name__ == '__main__':
#    plot_cammini()
#    MC_story_varying_N()
#    potential_term()
#    gap_energy()
    contiuum_E1_E0()
#    U_varying_T()
#    stato_fondamentale_singola()
