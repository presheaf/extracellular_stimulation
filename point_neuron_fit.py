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
# l5_models = glob(join(".", 'L5_NGC*'))
# l5_models = glob(join(".", 'L5_MC*'))
# l5_models = glob(join(".", 'L4_SS*'))
cell_folder = l5_models[0]
cell_name = cell_folder.split('/')[-1]
print cell_folder
nest.ResetKernel()
end_T = 200.                     # simulation time in ms


### Constants
##L5_TTPC2_cADpyr232_1
V_m = -65.36		# resting potential
C_m = 466.662		# capacitance of point neurons in picofarad
V_th = -50.		    # threshold potential of point neurons
tau_m = 25.077		# time constant of compartmental neurons
# tau_m = 3.8         # lower estimate of time constant because it seems to fit better
t_ref_m = 10.0

## L5_NGC_cNAC187_1
# V_m = -67.31   	# resting potential
# C_m = 49.54		# capacitance of point neurons in picofarad


## L5_SS_?
# V_m = -66.366  	# resting potential
# C_m = 181.2		# capacitance of point neurons in picofarad
# V_th = -48.85   # threshold potential of point neurons
# tau_m = 39.38	# time constant of compartmental neurons
# # tau_m = 0.476	# smaller time constant of compartmental neurons
# # tau_m = 19.38	# time constant of compartmental neurons
# t_ref_m = 5.     # testing refractory time



peak_widths = np.linspace(0.5, 20, 100)  # somehow relates to expected spike duration to look for
PEAK_DURATION = 0.0# duration from start to end of a spike


def soma_response(stim_amp, source_pos=[-70, 0, 0]):
    dt = 2**-4
    start_T = -200

    sigma = 0.3

    amp = stim_amp
    stim_start_time = 20
    T = end_T + stim_start_time
    
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
    # return t, cell.somav


def point_neuron(Ie, tau_m=tau_m, t_ref=t_ref_m):
    # print Ie
    nest.ResetKernel()

    # if param_dict is None:
    param_dict = {
        "I_e": Ie,    # injected current
        "V_th": V_th, # threshold
        "tau_m": tau_m, #tau, # membrane time constant
        "V_m": V_m, # initial membrane potential
        "V_reset": V_m, # reset potential
        "E_L": V_m, # resting potential
        "C_m": C_m, # membrane capacitance
        "t_ref": t_ref #refractory time
    }
    # else:
    #     param_dict["I_e"] = Ie
    #     param_dict["t_ref"] = t_ref
        
    n = nest.Create("iaf_psc_delta", params=param_dict)

    vm = nest.Create('voltmeter', params={'interval': 0.1})
    sd = nest.Create('spike_detector')
    
    nest.Connect(vm, n)
    nest.Connect(n, sd)

    nest.Simulate(end_T)
    vme = nest.GetStatus(vm, 'events')[0]
    t, V = vme['times'], vme['V_m']
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
    
    # def error2(pars):
    #     Vp = pars[0]
    #     model_U = Vp*(1 - np.exp(-s/tau_m))
    #     return np.linalg.norm(U - model_U)

    def error3(pars):
	    tau  = pars[0]
	    model_U = Vp_i*(1 - np.exp(-s/tau))
	    return np.linalg.norm(U - model_U)#, ord=np.inf)

    # choose tau s.t. model=data at voltage 0.8*Vp as initial guess
    f = 0.8
    tau_i = -s[np.argmin(np.abs(U/Vp_i - f))]/np.log(1-f)
    
    # tau, Vp = scipy.optimize.minimize(error, [tau_i, Vp_i]).x
    # return tau, Vp

    # Vp = scipy.optimize.minimize(error2, [Vp_i]).x
    # return tau_m, Vp[0]

    tau = scipy.optimize.minimize(error3, [tau_i]).x
    return tau[0], Vp_i



def fit_subtresh_point_I(comp_t, comp_V, fit_tau=True):
    """Computes necessary current into a point neuron to give a 
    response comparable to a non-spiking compartmental model."""

    if fit_tau:
        tau, Vp = RC_fit(comp_t, comp_V)
        return tau, C_m * Vp/tau
    else:
        comp_Vp = comp_V[-1] - comp_V[0]
        return C_m * comp_Vp/tau_m



def find_peaks(y):
    """Returns indices of peaks."""
    return scipy.signal.find_peaks_cwt(y, peak_widths)

    
def fit_supertresh_point_I(comp_t, comp_V, fit_t_ref=True):
    """Computes necessary current into a point neuron to give 
    a response comparable to a spiking compartmental model."""
    
    peak_inds = find_peaks(comp_V)
    spike_times = comp_t[peak_inds]
    if not fit_t_ref:
        ## assumes >1 spike
        w_comp = 1/(spike_times[1] - spike_times[0]) # spike frequency - assumes >1 spike
        if w_comp > 1/t_ref_m:                       # spike freq. is limited by t_ref
            w_comp = 0.99/t_ref_m 
        print spike_times[:2]
        print "w_comp", w_comp
        fit_I = C_m*(V_th - V_m)/(tau_m *(1- np.exp(t_ref_m/tau_m - 1/(tau_m*w_comp))))
        return fit_I
    else:
        t1 = spike_times[0]         # time first spike should occur - determines current
        fit_I = (V_th - V_m) * C_m/(tau_m*(1 - np.exp(-t1/tau_m)))
        t_ref = spike_times[1] - 2*t1 # refractory time making 2nd spike at right time
        return fit_I, t_ref
    
   
 
comp_I = -11000.
pos = [-50, 0, 0] #[-150, 50, 0]


if load_comp_data == "SAVE":
    comp_t, comp_V = soma_response(comp_I, pos)
    pickle.dump((comp_t, comp_V), open(pickle_fn, "wb"))
elif load_comp_data == "LOAD":
    comp_t, comp_V = pickle.load(open(pickle_fn, "rb"))
else:
    print "what is \"{}\"".format(load_comp_data)
    sys.exit(1)


# if (super_or_sub != "SUPER" and super_or_sub != "SUB"):
#     print "input SUPER or SUB, not \"{}\"".format(super_or_sub)
#     sys.exit(1)
    
# plt.plot(comp_t, comp_V)
# plt.title('Somatic membrane potential (compartmental model)')
# plt.ylabel('mV')
# plt.savefig(".png")

# peak_inds = find_peaks(comp_V)
# peak_ts = comp_t[peak_inds]
# peak_Vs = comp_V[peak_inds]
# plt.plot(peak_ts, peak_Vs, "ro", label="peaks")


# N = len(comp_t)
# plt.plot(comp_t[:N/10], comp_V[:N/10], label="comp model")
# plt.plot(comp_t[N/10], comp_V[N/10], "ro", label="{}".format(comp_V[N/10]))
# plt.legend(loc="lower right")

# print comp_V[0], comp_V[N/10]
# plt.savefig("comp_model_response.png")
# plt.close()
plt.plot(comp_t, comp_V, label="comp model")
# ax = plt.gca()
# ax.set_ylim([-65, 40])
# ax.set_xlim([-5, 200])
plt.savefig("comp_model_response.png")
# plt.plot(comp_t[N/10], comp_V[N/10], "ro", label="{}".format(comp_V[N/10]))
# sys.exit(1)

if super_or_sub == "SUB":
    fit_I = fit_subtresh_point_I(comp_t, comp_V, fit_tau=False)
    fit_t, fit_V = point_neuron(fit_I)
    print fit_I
    # fit_tau, fit_I = fit_subtresh_point_I(comp_t, comp_V, fit_tau=True)
    # fit_t, fit_V = point_neuron(fit_I, tau_m=fit_tau)
    # print fit_tau, fit_I

    plt.plot(fit_t, fit_V, label="fit point model")
    plt.legend(loc="lower right")
elif super_or_sub == "SUPER":
    # fit_I, t_ref = fit_supertresh_point_I(comp_t, comp_V)
    # fit_I, fit_t_ref = fit_supertresh_point_I(comp_t, comp_V, fit_t_ref=False)
    fit_I = fit_supertresh_point_I(comp_t, comp_V, fit_t_ref=False)
    fit_t, fit_V = point_neuron(fit_I)
    # print fit_t_ref, fit_I
    print fit_I
    plt.plot(fit_t, fit_V, label="fit point model")

    plt.legend(loc="upper right")
elif super_or_sub == "FUN":
    plt.close()

    # #SFN neuron
    # t_ref = 5. # refrac. time making firing rates of <200 Hz possible
    # V_m = -54. # resting potential
    # C_m = 87.349 # membrane capacitance of Gillies, Willshaw
    # tau_m = 12.8 # membrane time constant of Gillies, Willshaw
    # V_th = -48.7 # threshold potential
    # V_reset = -63.5 # reset potential
    # param_dict = {
    #     "V_reset": V_reset,
	# 	"E_L": V_m, 
    #     "V_m": V_reset, # initial membrane potential
	# 	"V_th": V_th, 
	# 	"tau_m": tau_m, 
	# 	"C_m": C_m, 
    # }
    
    # GPi neuron
    t_ref = 5. # ?
    V_m = -65. # resting potential
    C_m = 87.349 # ?
    tau_m = 12.8 # ?
    V_th = -48.7 # ?
    V_reset = -63.5 # ?
    param_dict = {
        "V_reset": V_reset,
		"E_L": V_m, 
        "V_m": V_reset, # initial membrane potential
		"V_th": V_th, 
		"tau_m": tau_m, 
		"C_m": C_m, 
    }
    R = tau_m/C_m
    t1 = 22.397 # necessary time of first spike for firing rate of 36.5 Hz
    ghost_I = 57.4552793226
    # fit_I =  (V_th-V_reset)/(R*(1-np.exp(-22.397/tau_m)))
    # fit_I = (V_th - V_reset)/(C_m * t1 /(1 + tau_m * t1))
    # fit_I = 26.4*fit_I
    # print fit_I
    # sys.exit(1)
    # fit_I= 0.
    # import IPython; IPython.embed()
    t, V = point_neuron(ghost_I, t_ref, param_dict=param_dict)
    plt.plot(t, V, label="point neuron")
    plt.plot([22.397], [V_reset], "ro")

    


plt.savefig("fit.png")
# print fit_V[0], comp_V[0]
