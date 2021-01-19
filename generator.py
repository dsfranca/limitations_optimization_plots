#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:09:29 2021

@author: danielstilckfranca
"""

import metropolis_basis as mt
from numpy import linalg as LA
import numpy as np
from import_sparse_maxcut import import_instance
#test
A=import_instance('G1.csv')
n=A.shape[0]
print("I am the new version! 50")
[eigs,v]=LA.eig(A.todense())
norm=max(np.abs(eigs))
schedule=np.linspace(0,1/norm,n)
b=np.zeros(n)
samples=50000
burn_in=int(5*n*np.log(n))
blob=mt.telescopic_product_external_smarter(A,b,schedule,samples,burn_in)
np.savetxt("schedule12.csv",schedule, delimiter=",")
np.savetxt("results12.csv",blob, delimiter=",")



def get_partitions_from_estimates(estimates):
    partitions=[]
    log_estimates=np.log(estimates)
    for k in range(0,len(estimates)):
        partitions.append(np.sum(log_estimates[0:k+1]))
    return partitions


def energy_estimates(partitions,schedule):
    energies=[]
    for k in range(1,len(partitions)):
        energies.append(-partitions[k]/schedule[k])
    return energies
        
        
#
mac_results=pd.read_csv('results12.csv',header=None)
mac_schedule=pd.read_csv('schedule12.csv',header=None)
mac_results=mac_results.values
mac_schedule=mac_schedule.values
#
#
#
partitions=get_partitions_from_estimates(mac_results)
#
np.savetxt("results_part12.csv", partitions, delimiter=",")
#
energies=energy_estimates(partitions,mac_schedule)

    
