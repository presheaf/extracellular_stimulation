from __future__ import division
#!/usr/bin/env python
'''
Test implementation using cell models of the Blue Brain Project with LFPy.
The example assumes that the complete set of cell models available from
https://bbpnmc.epfl.ch/nmc-portal/downloads is unzipped in this folder.
'''

import os
from os.path import join
import sys
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import neuron
import LFPy

neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")
neuron.load_mechanisms('mods')


def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    f = file("template.hoc", 'r')
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print 'template {} found!'.format(templatename)
            continue

    return templatename


def compile_all_mechanisms():
    """
    This function must probably be fixed.
    """
    #attempt to set up a folder with all unique mechanism mod files, compile, and
    #load them all
    if not os.path.isdir(os.path.join('mods')):
        print os.listdir('.')
        os.mkdir(os.path.join('mods'))

    neurons = glob(join('L5*'))
    print neurons
    
    for nrn in neurons:
        for nmodl in glob(os.path.join(nrn, 'mechanisms', '*.mod')):
            #print nmodl
            while not os.path.isfile(os.path.join('mods', os.path.split(nmodl)[-1])):
                os.system('cp {} {}'.format(nmodl, os.path.join('mods')))
               #break

    os.chdir('mods')
    os.system('nrnivmodl')


def return_cell(cell_folder, cell_name, end_T, dt, start_T):

    cwd = os.getcwd()
    os.chdir(cell_folder)
    print "Simulating ", cell_name

    f = file("template.hoc", 'r')
    templatename = get_templatename(f)
    f.close()

    f = file("biophysics.hoc", 'r')
    biophysics = get_templatename(f)
    f.close()

    f = file("morphology.hoc", 'r')
    morphology = get_templatename(f)
    f.close()

    print('Loading constants')
    neuron.h.load_file('constants.hoc')

    if not hasattr(neuron.h, morphology):
        neuron.h.load_file(1, "morphology.hoc")

    if not hasattr(neuron.h, biophysics):
        neuron.h.load_file(1, "biophysics.hoc")


    if not hasattr(neuron.h, templatename):
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    morphologyfile = os.listdir('morphology')[0]#glob('morphology\\*')[0]

    # Instantiate the cell(s) using LFPy
    cell = LFPy.TemplateCell(morphology=join('morphology', morphologyfile),
                     templatefile=os.path.join('template.hoc'),
                     templatename=templatename,
                     templateargs=0,
                     tstopms=end_T,
                     tstartms=start_T,
                     timeres_NEURON=dt,
                     timeres_python=dt,
                     # celsius=6,
                     v_init=-70)
    os.chdir(cwd)
    cell.set_rotation(z=np.pi/2, x=np.pi/2)
    return cell



def run_cell_model(cell_model, sim_folder, figure_folder, cell_model_id):

    np.random.seed(123 * cell_model_id)
    T = 1200
    dt = 2**-5

    cell_name = os.path.split(cell_model)[-1]
    cell = return_cell(cell_model, cell_name, T, dt)

    delay = 200
    stim_length = 1000
    weight = -0.23

    num_spikes = 0
    spikes = []
    cut_out = [2. / dt, 5. / dt]
    num_to_save = 10

    while not num_to_save <= num_spikes <= num_to_save * 3:
        if num_spikes >= num_to_save * 3:
            weight *= 0.75
        elif num_spikes < num_to_save:
            weight *= 1.25
        noiseVec, cell, syn = set_input(weight, dt, T, cell, delay, stim_length)

        cell.simulate(rec_imem=True)

        t = cell.tvec
        v = cell.somav
        t = t#[-cut_off_idx:] #- t[-cut_off_idx]
        v = v#[-cut_off_idx:]
        spikes = find_spike_idxs(v)
        num_spikes = len(spikes)
    plt.subplot(121)
    plt.plot(t, v)
    plt.plot(t[spikes], np.zeros(len(spikes)), 'o')
    plt.subplot(122)
    t = t[0:(cut_out[0] + cut_out[1])] - t[cut_out[0]]  
    v_spikes = np.zeros((num_to_save, len(t)))
    for idx, spike_idx in enumerate(spikes[:num_to_save]):
        v_spike = v[spike_idx - cut_out[0]:spike_idx + cut_out[1]]
        plt.plot(t, v_spike)

    plt.savefig(join(figure_folder, 'spike_%s.png' % cell_name))



def plot_morphology(cell_folder, cell_name, figure_folder):
    cell = return_cell(cell_folder, cell_name, end_T=1, start_T=0, dt=2**-3)
    i = 0
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[], aspect=1, xlim=[-300, 300], ylim=[-300, 1200])
    for sec in cell.allseclist:
        c = 'g' if 'axon' in sec.name() else 'k'
        for seg in sec:
            # if i == 0:
            #     ax.plot(cell.xmid[i], cell.zmid[i], 'o', c=c, ms=10)
            # else:
            ax.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]], c=c, clip_on=False)
            i += 1
    ax.plot([200, 300], [-100, -100], lw=3, c='b')
    ax.text(200, -160, '100 $\mu$m')
    fig.savefig(join(figure_folder, 'morph_%s.png' % cell_name))


def plot_cell_soma_response(cell_folder, cell_name, figure_folder):
    cell = return_cell(cell_folder, cell_name, end_T=100, start_T=-500, dt=2**-3)

    synapse_parameters = {
        'idx' : cell.get_closest_idx(x=-200., y=0., z=800.),
        'e' : 0.,                   # reversal potential
        'syntype' : 'ExpSyn',       # synapse type
        'tau' : 2.,                 # synaptic time constant
        'weight' : .01,            # synaptic weight
        'record_current' : True,    # record synapse current
    }

    # Create synapse and set time of synaptic input
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([20.]))
    cell.simulate()

    i = 0
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(121, frameon=False, xticks=[], yticks=[], aspect=1, xlim=[-300, 300], ylim=[-300, 1200])
    for sec in cell.allseclist:
        c = 'g' if 'axon' in sec.name() else 'k'
        for seg in sec:
            ax.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]], c=c, clip_on=False)
            i += 1
    ax.plot([200, 300], [-100, -100], lw=3, c='b')
    ax.text(200, -160, '100 $\mu$m')

    ax = fig.add_subplot(122, title='Somatic membrane potential')
    ax.plot(cell.tvec, cell.somav)

    fig.savefig(join(figure_folder, 'single_synapse_%s_%d.png' % (cell_name, neuron.h.celsius)))



if __name__ == '__main__':

    compile_all_mechanisms()
    
    
    # plot_cell_soma_response(join('cell_models', 'L5_NBC_cACint209_1'), 'L5_NBC_cACint209_1', '.')
    sys.exit()
    if len(sys.argv) > 1:
        print sys.argv
        exec('%s("%s", "%s", "%s")' % (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
   


