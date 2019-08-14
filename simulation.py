import os
import sys
import csv
import nest
import pylab as pl
import numpy as np
import thorns as th
import pyspike as spk
import cPickle as pickle
from mogutda import SimplicialComplex

###############################################################################
def getstatus(neuron):
    items = nest.GetStatus(neuron)[0].items()
    for i in range(len(items)):
        print(str(items[i][0]) + " = " + str(items[i][1]))

def matrix_from_csv(matrixfile):
    with open("connection_matrices/"+matrixfile) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        matrix = list(readCSV)
    return(matrix)

def simplist_from_matrix(matrixfile):
    os.chdir("neurotop/src")
    os.system("./directed ../../connection_matrices/" + matrixfile +".csv 1 1")
    os.chdir("../../connection_matrices")
    os.rename(matrixfile+".csv_simplices_dimension_.txt", matrixfile+"_dlist.csv")
    os.chdir("../")

def mogulist_from_simplist(simpfile):
    simpfile = simpfile + "_dlist.csv"
    simps = []
    for line in open("connection_matrices/" + simpfile):
        line = line.rstrip()
        line = line.split(" ")
        for num in range(len(line)):
            line[num] = int(line[num])

        simps.append(tuple(line))

    return(simps)

def change_homology(neuron):
    return(True)

###############################################################################

matrix_name = sys.argv[1]
simplist_from_matrix(matrix_name)     # creates list of simplices from matrix using neurtop

M = matrix_from_csv(matrix_name + ".csv")   # connection matrix

ablationlist = [4]
# ablationlist = [4,6,8,14,31,3,28,21]

# simulation parameters
dt = 0.01
duration = 200
runs = len(ablationlist)+1

# model parameters
N = len(M[0])
g = 20000.0                     # synapse conductance


V_m = 62.0                     #	- Membrane potential in mV
E_L = -54.402                  #	- Resting membrane potential in mV.
g_L = 30.0                     #	- Leak conductance in nS.
C_m	= 100.0                    #	- Capacity of the membrane in pF.
tau_syn_ex = 0.2               #	- Rise time of the excitatory synaptic alpha function in ms.
tau_syn_in = 2.0               #	- Rise time of the inhibitory synaptic alpha function in ms.
E_Na = 50.0                    #	- Sodium reversal potential in mV.
g_Na = 12000.0                 #	- Sodium peak conductance in nS.
E_K	 = -77.0                   #	- Potassium reversal potential in mV.
g_K	 = 3600.0                  #	- Potassium peak conductance in nS.
Act_m = 0.0540786107908        #	- Activation variable m
Act_h = 0.447264612753         #	- Activation variable h
Inact_n = 0.402688712673       #	- Inactivation variable n
I_e	 = 690.0                   #	- Constant external input current in pA.

paramdict = {"V_m": V_m, "E_L": E_L, "g_L": g_L, "C_m": C_m, "tau_syn_ex": tau_syn_ex,
              "tau_syn_in": tau_syn_in, "E_Na": E_Na, "g_Na": g_Na, "E_K": E_K, "g_K": g_K, "Act_m": Act_m,
              "Act_h": Act_h, "Inact_n": Inact_n, "I_e": I_e}


nest.ResetKernel()
nest.SetKernelStatus({"resolution": dt, "use_wfr": False,
                      "print_time": True,
                      "overwrite_files": True})


neurons = []
for i in range(N):
    neurons.append(nest.Create("hh_psc_alpha", params=paramdict))


# for i in range(N):
#     nest.SetStatus(neurons[i], {"V_m": -62.0 + np.random.random()*15})
for i in range(N):
    nest.SetStatus(neurons[i], {"V_m": -62.0 - i*3})

Mult = []
for i in range(N):
    Mult.append(nest.Create("multimeter"))
    nest.SetStatus(Mult[i], {"withtime":True, "record_from":["V_m"], "interval":dt})
    nest.Connect(Mult[i], neurons[i])


sd = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
for i in range(N):
    nest.Connect(neurons[i],sd)

# connect neurons ##############################################################
nest.CopyModel("static_synapse", "inhibitory", {"weight": (-1)*g})
nest.CopyModel("static_synapse", "excitatory", {"weight": g})

num_synapse = 0
for i in range(N):
    for j in range(N):
        if(M[i][j] == "1"):
            # nest.Connect(neurons[i],neurons[j], syn_spec="inhibitory")
            nest.Connect(neurons[i],neurons[j], syn_spec="excitatory")
            num_synapse += 1
            # print("connecting neuron " + str(i) + " to " + str(j))

# measure original homology of network #########################################
simps = mogulist_from_simplist(matrix_name)
SC = SimplicialComplex(simplices=simps)
bettis = [SC.betti_number(0),SC.betti_number(1),SC.betti_number(2),SC.betti_number(3),SC.betti_number(4),SC.betti_number(5),SC.betti_number(6)]
print(bettis)

proceed = raw_input("proceed? ")
if(proceed != "y"):
    print(asdasd)

# run the simulation ###########################################################
# once to stabilize the network
nest.Simulate(duration)

# now start ablating
for i in range(runs-1):
    for key in paramdict:
        try:
            nest.SetStatus(neurons[ablationlist[i]], {key: 0.0})
        except:
            if(key == "C_m"):
                nest.SetStatus(neurons[ablationlist[i]], {key: 100000000.0})

    print("ablating neuron " + str(ablationlist[i]))
    print("simulating " + str(i) + " out of " + str(runs-1))
    nest.Simulate(duration)


# plot that shit ###############################################################

dmm = []
Vms = []
ts = []
for i in range(N):
    dmm.append(nest.GetStatus(Mult[i])[0])
    Vms.append(dmm[i]["events"]["V_m"])
    ts.append(dmm[i]["events"]["times"])

for i in range(len(Vms)):
    pl.plot(ts[i],Vms[i])

pl.xlim(-5,runs*duration)
pl.ylim(-85,75)
pl.show(block=False)


dSD = nest.GetStatus(sd,keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]
spiketimes = []
for i in range(len(evs)):
    spiketimes.append((evs[i],ts[i]))

spiketimes = sorted(spiketimes, key=lambda tup: tup[0])
spikes = []
for i in range(N):
    spikes.append([])

for i in range(len(spiketimes)):
    spikes[spiketimes[i][0]-1].append(float(spiketimes[i][1]))


pl.figure(2)
pl.plot(ts, evs, "|")
for i in range(1,len(ablationlist)+1):
    pl.axvline(x=duration*i, linewidth=0.2, color="b")
    pl.plot(duration*i, ablationlist[i-1]+1, color="b", marker="x")


pl.ylim(0,N+1)
pl.xlim(0,duration*runs)
pl.show(block=False)



########## measure synchrony ##################################################
slices = []
for run in range(runs):
    section = []
    for n in range(N):
        section.append([])
        subint = [x for x in spikes[n] if x >= ((run-1)*duration) and x <= (run*duration)]
        section[n] = spk.SpikeTrain(subint, (0,duration))

    slices.append(section)

pl.figure(3)

sync = []
for c in range(len(slices)):
    # sync.append(np.var(spk.spike_sync_matrix(slices[c])))
    sync.append(np.linalg.norm(spk.spike_sync_matrix(slices[c])))
    # sync.append(np.sum(spk.spike_sync_matrix(slices[c])))

pl.plot(sync, linestyle="-", marker="o", markersize="7")
# pl.hlines(15, 0, len(homXS), linewidth=0.3)
pl.grid(which='both', axis='y')
# pl.xlim(xmin=-0.5,xmax=len(homXS)+1.5)
for i in range(len(sync)):
    pl.text(i, sync[i]-0.3, str(round(sync[i],2)))

pl.show()
