# -*- coding: utf-8 -*-
#
# brunel-delta-nest.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

# This version uses NEST's Connect functions.

import nest

import time
from numpy import exp
import pandas as pd
import numpy as np

def sim_brunel_delta(dt=0.1,
                     simtime=1000.0,
                     delay=1.5,
                     g=5.0,
                     eta=2.0,
                     epsilon=0.1,
                     order=2500,
                     J=0.1,
                     V_reset=0.0,
                     input_stop=np.inf,
                     N_rec=50,
                     num_threads=1,
                     print_report=True):

    nest.ResetKernel()
    nest.set_verbosity('M_WARNING')
    
    startbuild = time.time()

    NE        = 4*order
    NI        = 1*order
    N_neurons = NE+NI

    CE    = int(epsilon*NE) # number of excitatory synapses per neuron
    CI    = int(epsilon*NI) # number of inhibitory synapses per neuron  
    C_tot = int(CI+CE)      # total number of synapses per neuron

    # Initialize the parameters of the integrate and fire neuron
    tauMem = 20.0
    theta  = 20.0

    J_ex  = J
    J_in  = -g*J_ex

    nu_th  = theta/(J*CE*tauMem)
    nu_ex  = eta*nu_th
    p_rate = 1000.0*nu_ex*CE

    nest.SetKernelStatus({"resolution": dt, "print_time": True,
                          "local_num_threads": num_threads})

    if print_report:
        print("Building network")

    neuron_params= {"C_m":        1.0,
                    "tau_m":      tauMem,
                    "t_ref":      2.0,
                    "E_L":        0.0,
                    "V_reset":    V_reset,
                    "V_m":        0.0,
                    "V_th":       theta}


    nest.SetDefaults("iaf_psc_delta", neuron_params)

    nodes_ex=nest.Create("iaf_psc_delta",NE)
    nodes_in=nest.Create("iaf_psc_delta",NI)

    nest.SetDefaults("poisson_generator",{"rate": p_rate, 'stop': input_stop})
    noise=nest.Create("poisson_generator")

    espikes=nest.Create("spike_detector")
    ispikes=nest.Create("spike_detector")

    nest.SetStatus(espikes,[{"label": "brunel-py-ex",
                       "withtime": True,
                       "withgid": True,
                       "to_file": False}])

    nest.SetStatus(ispikes,[{"label": "brunel-py-in",
                       "withtime": True,
                       "withgid": True,
                       "to_file": False}])

    if print_report:
        print("Connecting devices")

    nest.CopyModel("static_synapse","excitatory",{"weight":J_ex, "delay":delay})
    nest.CopyModel("static_synapse","inhibitory",{"weight":J_in, "delay":delay})

    nest.Connect(noise,nodes_ex, syn_spec="excitatory")
    nest.Connect(noise,nodes_in, syn_spec="excitatory")

    nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory")
    nest.Connect(nodes_in[:N_rec], ispikes, syn_spec="excitatory")

    if print_report:
        print("Connecting network")

    if print_report:
        print("Excitatory connections")

    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
    nest.Connect(nodes_ex, nodes_ex+nodes_in, conn_params_ex, "excitatory")

    if print_report:
        print("Inhibitory connections")

    conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
    nest.Connect(nodes_in, nodes_ex+nodes_in, conn_params_in, "inhibitory")

    endbuild=time.time()

    if print_report:
        print("Simulating")

    nest.Simulate(simtime)

    endsimulate= time.time()

    if print_report:
        events_ex = nest.GetStatus(espikes,"n_events")[0]
        rate_ex   = events_ex/simtime*1000.0/N_rec
        events_in = nest.GetStatus(ispikes,"n_events")[0]
        rate_in   = events_in/simtime*1000.0/N_rec

        num_synapses = nest.GetDefaults("excitatory")["num_connections"]+\
        nest.GetDefaults("inhibitory")["num_connections"]

        build_time = endbuild-startbuild
        sim_time   = endsimulate-endbuild

        print("Brunel network simulation (Python)")
        print("Number of neurons : {0}".format(N_neurons))
        print("Number of synapses: {0}".format(num_synapses))
        print("       Exitatory  : {0}".format(int(CE * N_neurons) + N_neurons))
        print("       Inhibitory : {0}".format(int(CI * N_neurons)))
        print("Excitatory rate   : %.2f Hz" % rate_ex)
        print("Inhibitory rate   : %.2f Hz" % rate_in)
        print("Building time     : %.2f s" % build_time)
        print("Simulation time   : %.2f s" % sim_time)

    exc_spikes = nest.GetStatus(espikes, 'events')[0]
    inh_spikes = nest.GetStatus(ispikes, 'events')[0]

    return pd.DataFrame(exc_spikes), pd.DataFrame(inh_spikes)


def build_brunel_delta_plastic(dt=0.1,
                     delay=1.5,
                     g=5.0,
                     eta=2.0,
                     epsilon=0.1,
                     order=2500,
                     J=0.1,
                     alpha=2.02,
                     lambd=0.01,
                     Wmax=3.,
                     V_reset=0.0,
                     input_stop=np.inf,
                     N_rec=50,
                     num_threads=1,
                     print_report=True):

    nest.ResetKernel()
    nest.set_verbosity('M_WARNING')
    
    startbuild = time.time()

    NE        = 4*order
    NI        = 1*order
    N_neurons = NE+NI

    CE    = int(epsilon*NE) # number of excitatory synapses per neuron
    CI    = int(epsilon*NI) # number of inhibitory synapses per neuron  
    C_tot = int(CI+CE)      # total number of synapses per neuron

    # Initialize the parameters of the integrate and fire neuron
    tauMem = 20.0
    theta  = 20.0

    J_ex  = J
    J_in  = -g*J_ex

    nu_th  = theta/(J*CE*tauMem)
    nu_ex  = eta*nu_th
    p_rate = 1000.0*nu_ex*CE

    nest.SetKernelStatus({"resolution": dt, "print_time": True,
                          "local_num_threads": num_threads})

    if print_report:
        print("Building network")

    neuron_params= {"C_m":        1.0,
                    "tau_m":      tauMem,
                    "t_ref":      2.0,
                    "E_L":        0.0,
                    "V_reset":    V_reset,
                    "V_m":        0.0,
                    "V_th":       theta}


    nest.SetDefaults("iaf_psc_delta", neuron_params)

    nodes_ex=nest.Create("iaf_psc_delta",NE)
    nodes_in=nest.Create("iaf_psc_delta",NI)

    nest.SetStatus(nodes_ex+nodes_in, 'V_m', np.random.uniform(low=-20., high=20., size=(NE+NI,)))

    nest.SetDefaults("poisson_generator",{"rate": p_rate, 'stop': input_stop})
    noise=nest.Create("poisson_generator")

    espikes=nest.Create("spike_detector")
    ispikes=nest.Create("spike_detector")

    nest.SetStatus(espikes,[{"label": "brunel-py-ex",
                       "withtime": True,
                       "withgid": True,
                       "to_file": False}])

    nest.SetStatus(ispikes,[{"label": "brunel-py-in",
                       "withtime": True,
                       "withgid": True,
                       "to_file": False}])

    if print_report:
        print("Connecting devices")

    nest.CopyModel("stdp_synapse","excitatory_plastic",{'alpha': alpha, 'lambda': lambd, 'Wmax': Wmax,
                                                        "delay":delay})
    nest.CopyModel("static_synapse","excitatory",{"weight":J_ex, "delay":delay})
    nest.CopyModel("static_synapse","inhibitory",{"weight":J_in, "delay":delay})

    nest.Connect(noise,nodes_ex, syn_spec="excitatory")
    nest.Connect(noise,nodes_in, syn_spec="excitatory")

    nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory")
    nest.Connect(nodes_in[:N_rec], ispikes, syn_spec="excitatory")

    if print_report:
        print("Connecting network")

    if print_report:
        print("Excitatory connections")

    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
    nest.Connect(nodes_ex, nodes_ex, conn_params_ex, {'model': 'excitatory_plastic',
                                                      'weight': {'distribution': 'uniform',
                                                                 'low': 0.5 * J_ex,
                                                                 'high': 1.5 * J_ex}})
    nest.Connect(nodes_ex, nodes_in, conn_params_ex, {'model': 'excitatory',
                                                      'weight': {'distribution': 'uniform',
                                                                 'low': 0.5 * J_ex,
                                                                 'high': 1.5 * J_ex}})

    if print_report:
        print("Inhibitory connections")

    conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
    nest.Connect(nodes_in, nodes_ex+nodes_in, conn_params_in, "inhibitory")

    endbuild=time.time()

    if print_report:
        num_synapses = sum(nest.GetDefaults(syn_model)["num_connections"]
                           for syn_model in ('excitatory', 'excitatory_plastic',
                                             'inhibitory'))

        build_time = endbuild-startbuild

        print("Brunel network (Python)")
        print("Number of neurons : {0}".format(N_neurons))
        print("Number of synapses: {0}".format(num_synapses))
        print("       Exitatory  : {0}".format(int(CE * N_neurons) + N_neurons))
        print("       Inhibitory : {0}".format(int(CI * N_neurons)))
        print("Building time     : %.2f s" % build_time)

    return espikes, ispikes, nodes_ex, nodes_in



def sim_brunel_delta_plastic(simtime,
                             ex_spike_det, in_spike_det,
                             nodes_ex_wgt=None, nodes_in_wgt=None,
                             print_report=True):

    nodes_ex_wgt = list(nodes_ex_wgt) if nodes_ex_wgt is not None else []
    nodes_in_wgt = list(nodes_in_wgt) if nodes_in_wgt is not None else []
    
    startsimulate=time.time()

    if print_report:
        print("Simulating")

    nest.Simulate(simtime)

    endsimulate= time.time()

    if print_report:
        sim_time   = endsimulate-startsimulate

        print("Brunel network simulation (Python)")
        print("Simulation time   : %.2f s" % sim_time)

    exc_spikes = nest.GetStatus(ex_spike_det, 'events')[0]
    inh_spikes = nest.GetStatus(in_spike_det, 'events')[0]

    w_pl = nest.GetStatus(nest.GetConnections(source=nodes_ex_wgt+nodes_in_wgt, synapse_model='excitatory_plastic'), 'weight')
    
    return pd.DataFrame(exc_spikes), pd.DataFrame(inh_spikes), np.array(w_pl)
