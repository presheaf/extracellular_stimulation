import LFPy
import sys
load_comp_data = sys.argv[1]    # SAVE or LOAD
pickle_fn = sys.argv[2]         
super_or_sub = sys.argv[3]      # SUPER or SUB

import os, nest, scipy, pickle, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from glob import glob

# matplotlib.use('AGG')
from plotting_convention import *

hbp_cells = '.'

sys.path.append(hbp_cells)
from hbp_cells import return_cell

l5_models = glob(join(".", 'L5_TTPC2*'))
cell_folder = l5_models[0]
cell_name = cell_folder.split('/')[-1]
print cell_folder

## Constants

V_m = -65.36		# resting potential of L5_TTPC2_cADpyr232_1
C_m = 466.662		# capacitance of point neurons in picofarad
V_th = -50.		    # threshold potential of point neurons
tau_m = 25.077		# time constant of compartmental neurons
tau_m = 8.0         # lower estimate of time constant because it seems to give better fit

peak_widths = np.linspace(0.5, 20, 100)  # somehow relates to expected spike duration to look for
PEAK_DURATION = 0.0# duration from start to end of a spike


def soma_response(stim_amp, source_pos=[-70, 0, 0]):
    dt = 2**-4
    T = 100
    start_T = -200

    sigma = 0.3

    amp = stim_amp
    stim_start_time = 20

    n_tsteps = int(T / dt + 1)
    t = np.arange(n_tsteps) * dt

    sources_x = np.array([source_pos[0]])
    sources_y = np.array([source_pos[1]])
    sources_z = np.array([source_pos[2]])

    def ext_field(x, y, z):
	ef = 0
	for s_idx in range(len(sources_x)):
	    ef += 1 / (4 * np.pi * sigma * np.sqrt((sources_x[s_idx] - x) ** 2 +
						   (sources_y[s_idx] - y) ** 2 +
						   (sources_z[s_idx] - z) ** 2))
	return ef

    cell = return_cell(cell_folder, cell_name, T, dt, start_T)

    pulse = np.zeros(n_tsteps)
    start_time_idx = np.argmin(np.abs(t - stim_start_time))
    pulse[start_time_idx:] = amp

    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps))
    v_cell_ext[:, :] = ext_field(cell.xmid, cell.ymid, cell.zmid).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps)

    cell.insert_v_ext(v_cell_ext, t)
    print("running simulation...")
    cell.simulate(rec_imem=False, rec_isyn=True, rec_vmem=True)
    np.save('vmem_cell.npy', cell.vmem)
    vmem = cell.vmem
    # print '[', amp, ',', np.max(cell.somav) - cell.somav[start_time_idx - 1], '],'

    # change time coordinates so that stimulation starts at t=0
    return (t[start_time_idx-1:] - stim_start_time), cell.somav[start_time_idx-1:]


def point_neuron(Ie, t_ref=25.0):
    # print Ie
    nest.ResetKernel()
    
    n = nest.Create("iaf_psc_delta", params={"I_e": Ie,    # injected current
					                         "V_th": V_th, # threshold
					                         "tau_m": tau_m, #tau, # membrane time constant
					                         "V_m": V_m, # membrane potential?
                                             "V_reset": V_m, # reset potential
					                         "E_L": V_m, # resting potential
					                         "C_m": C_m, # membrane capacitance
                                             "t_ref": t_ref}) #refractory time



    vm = nest.Create('voltmeter', params={'interval': 0.1})
    sd = nest.Create('spike_detector')
    
    nest.Connect(vm, n)
    nest.Connect(n, sd)

    nest.Simulate(80)
    vme = nest.GetStatus(vm, 'events')[0]
    t, V = vme['times'], vme['V_m']
    # V = V - V[0] + (-65.359)	# dirty hack to make resting potential -65.4 mV
    return t, V


def RC_fit(t, V):
    # fits RC circuit response by minimizing squared error
    
    V0 = V[0]			# resting potential
    t0 = t[0]			# start of stimulation

    U = V - V0
    s = t - t0
    Vp_i = U[-1]
    # Vp_i = max(U)
    
    # def error(pars):
	#     tau, Vp = pars
	#     model_U = Vp*(1 - np.exp(-s/tau))
	#     return np.linalg.norm(U - model_U)#, ord=np.inf)
    
    def error2(pars):
        Vp = pars[0]
        model_U = Vp*(1 - np.exp(-s/tau_m))
        return np.linalg.norm(U - model_U)

    # choose tau s.t. model=data at voltage 0.8*Vp as initial guess
    # f = 0.8
    # tau_i = -s[np.argmin(np.abs(U/Vp_i - f))]/np.log(1-f)
    # tau, Vp = scipy.optimize.minimize(error, [tau_i, Vp_i]).x
    # # Vp -= V0
    # return tau, Vp
    Vp = scipy.optimize.minimize(error2, [Vp_i]).x
    return Vp[0]



def fit_subtresh_point_I(comp_t, comp_V):
    """Computes necessary current into a point neuron to give a 
    response comparable to a non-spiking compartmental model."""

    comp_Vp = comp_V[-1] - comp_V[0]
    return C_m * comp_Vp/tau_m


def find_peaks(y):
    """Returns indices of peaks."""
    return scipy.signal.find_peaks_cwt(y, peak_widths)

    
def fit_supertresh_point_I(comp_t, comp_V):
    """Computes necessary current into a point neuron to give 
    a response comparable to a spiking compartmental model."""
    
    peak_inds = find_peaks(comp_V)
    spike_times = comp_t[peak_inds]
    t1 = spike_times[0]
    
    t_ref = spike_times[1] - t1 # OOPS: assumes >1 spike, should fix
    fit_I = - V_th * C_m/(tau_m*(1 - np.exp(-t1/tau_m)))
    return fit_I, t_ref
    
    

comp_I = -20000.0
pos = [-150, 50, 0]


if load_comp_data == "SAVE":
    comp_t, comp_V = soma_response(comp_I, pos)
    pickle.dump((comp_t, comp_V), open(pickle_fn, "wb"))
elif load_comp_data == "LOAD":
    comp_t, comp_V = pickle.load(open(pickle_fn, "rb"))
else:
    print "what is \"{}\"".format(load_comp_data)
    sys.exit(1)


if (super_or_sub != "SUPER" and super_or_sub != "SUB"):
    print "input SUPER or SUB, not \"{}\"".format(super_or_sub)
    sys.exit(1)
    
# plt.plot(comp_t, comp_V)
# plt.title('Somatic membrane potential (compartmental model)')
# plt.ylabel('mV')
# plt.savefig(".png")

# peak_inds = find_peaks(comp_V)
# peak_ts = comp_t[peak_inds]
# peak_Vs = comp_V[peak_inds]
# plt.plot(peak_ts, peak_Vs, "ro", label="peaks")



plt.plot(comp_t, comp_V, label="comp model")

if super_or_sub == "SUB":
    fit_I = fit_subtresh_point_I(comp_t, comp_V)
    fit_t, fit_V = point_neuron(fit_I)
    plt.plot(fit_t, fit_V, label="fit point model")
    plt.legend(loc="lower right")
elif super_or_sub == "SUPER":
    fit_I, t_ref = fit_supertresh_point_I(comp_t, comp_V)
    fit_t, fit_V = point_neuron(fit_I, t_ref)
    plt.plot(fit_t, fit_V, label="fit point model")
    plt.legend(loc="upper right")

plt.savefig("fit.png")

