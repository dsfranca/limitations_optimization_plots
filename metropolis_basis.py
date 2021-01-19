#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:42:17 2020

@author: danielstilckfranca
"""
from scipy.sparse import csr_matrix, find
import random
import numpy as np
import time
import itertools
import pandas as pd

#be careful with factor of 2
    
def random_graph(n,delta):
    rows=np.zeros(delta*n,dtype=int)
    
    for k in range(0,n):
        for l in range(0,delta):
            rows[k*delta+l]=int(k)
    columns=np.random.randint(n, size=len(rows))
    for k in range(0,n):
        for l in range(0,delta):
            if columns[k*delta+l]==k:
                columns[k*delta+l]+=1

    rows_trans=np.concatenate((rows,columns),axis=None)
    columns_trans=np.concatenate((columns,rows),axis=None)
   
    A=csr_matrix((0.5*np.ones(2*delta*n), (rows_trans, columns_trans)))

    return A
            
#adjacency of line of length n
def generate_line(n):
    rows=[]
    columns=[]
    for k in range(0,n-1):
        rows.append(k)
        
        columns.append(k+1)
    rows_trans=np.concatenate((rows,columns),axis=None)
    columns_trans=np.concatenate((columns,rows),axis=None)
    
    A=csr_matrix((0.5*np.ones(len(rows_trans)), (rows_trans, columns_trans)))
    
    return A


def generate_line_random(n):
    rows=[]
    columns=[]
    for k in range(0,n-1):
        rows.append(k)
        
        columns.append(k+1)
    rows_trans=np.concatenate((rows,columns),axis=None)
    columns_trans=np.concatenate((columns,rows),axis=None)
    A=csr_matrix((np.random.normal(0,0.25,size=len(rows_trans)), (rows_trans, columns_trans)))
    A=A+np.transpose(A)
    
    return A
#for sanity checks, generates n/2 independent pairs of ZZ
def generate_independent_ZZ(n):
    rows=[]
    columns=[]
    for k in range(0,int(n/2)):
        rows.append(2*k)
        
        columns.append(2*k+1)
    rows_trans=np.concatenate((rows,columns),axis=None)
    columns_trans=np.concatenate((columns,rows),axis=None)
    
    A=csr_matrix((0.5*np.ones(len(rows_trans)), (rows_trans, columns_trans)))
    
    return A

def metropolis(beta,A,times,current):
    n=A.shape[0]
    
    for k in range(0,times):
        current=update(current,beta,A)
    

    return current

def gen_samples_metropolis(beta,A,b,samples,burn_in):
    n=A.shape[0]
    outputs=[]
    for m in range(0,samples):
        current=2*np.random.randint(2,size=n)-np.ones(n)
        blob=A.dot(current)
        energy=current.dot(blob)
        for  k  in range(0,burn_in):
            #print("Before",[current,energy])
            [current,energy]=update_external(current,beta,A,b,energy)
            #print("After",[current,energy])
        outputs.append(current)
    
    return outputs
    
    
def gen_samples_metropolis_repeat(beta,A,b,samples,burn_in):
    n=A.shape[0]
    outputs=[]
    first=1
    for m in range(0,samples):
        if first==1:
            current=2*np.random.randint(2,size=n)-np.ones(n)
            blob=A.dot(current)
            energy=current.dot(blob)
            
        for  k  in range(0,burn_in):
                
            [current,energy]=update_external(current,beta,A,b,energy)
            #print("After",[current,energy])
        glue=current.copy()
        first=0
        burn_in=5
        outputs.append(glue)

        

    return outputs
    
    
    
    
    
#computes the energy of a vector current from a Hamiltonian A
def compute_energy(current,A):
    current_trans=np.transpose(current)
    old_energy=current_trans.dot(A)
    old_energy=old_energy.dot(current)


#computes one update of Metropolis-Hastings given the current guess 
def update(current,beta,A,current_energy):
    

    new_energy=current_energy
    x=random.randint(0,n-1)  
    sum_neigh=0
    B=A[x,:].nonzero()
    for k in range(0,len(B[1])):
        sum_neigh+=A[x,B[1][k]]*current[B[1][k]]
    
    change_energy=2*current[x]*sum_neigh
    print(sum_neigh,change_energy)
    if change_energy<0:
        #print("acccepted smaller")
        current[x]=(-1)*current[x]
        new_energy=current_energy+change_energy
    else:
        p=random.random()
        if np.exp(-beta*(change_energy))>p:
            current[x]=(-1)*current[x]
            new_energy=current_energy+change_energy
            #print("acccepted larger")
        
    
    return [current,new_energy]


#does the update with additional external field b
def update_external(current,beta,A,b,current_energy):
    
    n=A.shape[0]
    new_energy=current_energy
    x=random.randint(0,n-1)  
    sum_neigh=0
    B=A[x,:].nonzero()
    for k in range(0,len(B[1])):
        sum_neigh+=A[x,B[1][k]]*current[B[1][k]]
    
    change_energy=-4*current[x]*sum_neigh-4*current[x]*b[x]
    #print(current,x,current[x],2*current[x]*sum_neigh)
    if change_energy<0:
        current[x]=(-1)*current[x]
        new_energy=current_energy+change_energy
    else:
        p=random.random()
        if np.exp(-beta*(change_energy))>p:
            current[x]=(-1)*current[x]
            new_energy=current_energy+change_energy
        
    
    return [current,new_energy]


def telescopic_product_external(A,b,schedule,samples,burn_in):
    estimates_ratio=[]
    n=A.shape[0]
    for ratio in range(0,len(schedule)-1):
        print("estimate number",ratio)
        
        current_ratio=0
        for m in range(0,samples):
            start = time.time()
            print("samples",m,ratio)
            current=2*np.random.randint(2,size=n)-np.ones(n)
            blob=A.dot(current)
            energy=current.dot(blob)+current.dot(b)
            for  k  in range(0,burn_in):
                [current,energy]=update_external(current,schedule[ratio],A,b,energy)
            end=time.time()
            print("it took:",end-start)
            current_ratio+=np.exp(-(schedule[ratio+1]-schedule[ratio])*energy)
        estimates_ratio.append(current_ratio/samples)
        
            
    return estimates_ratio
    
def telescopic_product_external_smarter(A,b,schedule,samples,burn_in):
    estimates_ratio=[]
    n=A.shape[0]
    first=1
    for ratio in range(0,len(schedule)-1):
        start = time.time()
        print("estimate number",ratio)
        
        current_ratio=0
        for m in range(0,samples):
            
            
            
            if first==1:
                current=2*np.random.randint(2,size=n)-np.ones(n)
                blob=A.dot(current)
                energy=current.dot(blob)+current.dot(b)
            if first<1:
                burn_in=5
            first=0
            for  k  in range(0,burn_in):
                [current,energy]=update_external(current,schedule[ratio],A,b,energy)
            end=time.time()
            
            current_ratio+=np.exp(-(schedule[ratio+1]-schedule[ratio])*energy)
        estimates_ratio.append(current_ratio/samples)
        end=time.time()
        print("it took:",end-start)
            
    return estimates_ratio

#computes the partition function brute force
def partition(A,beta):
    start = time.time()
    n=A.shape[0]
    estimate=0
    for initial in itertools.product([-1, 1], repeat=n):
        y=initial
        
        initial=np.array(initial)
        new_energy=A.dot(initial)
        #print(new_energy.shape,initial.shape)
        new_energy=initial.dot(new_energy)
        estimate=estimate+np.exp(-beta*new_energy)
        print(initial,new_energy,estimate/(2**n))
     
            
    end = time.time()
    return estimate/(2**n)

def find_min(A):
    start = time.time()
    n=A.shape[0]
    estimate=100
    for initial in itertools.product([-1, 1], repeat=n):
        
        initial=np.array(initial)
        new_energy=A.dot(initial)
        #print(new_energy.shape,initial.shape)
        new_energy=initial.dot(new_energy)
        if new_energy<estimate:
            estimate=new_energy
     
            
    end = time.time()
    return estimate

def get_partitions_from_estimates(estimates):
    partitions=[]
    log_estimates=np.log(estimates)
    for k in range(0,len(estimates)):
        partitions.append(np.sum(log_estimates[0:k+1]))
    return partitions
        
#computes the value of Pauli i,j
def pauli_zz(state,i,j):
    if (state[i]*state[j]>0):
        
        return 1
        
    else: 
        return -1

def pauli_z(state,i):
    if (state[i]>0):
        return -1
    else: 
        return 1


def generate_histogram(beta,A,b,samples,burn_in):
    arr=gen_samples_metropolis(beta,A,b,samples,burn_in)
        # initializing dict to store frequency of each element
    n=A.shape[0]
    states=[]
    frequencies=[]
    for initial in itertools.product([-1, 1], repeat=n):
        y=initial
        print(y)
        states.append(initial)
        initial=np.array(initial)
        frequency=0
        for k in range(0,len(arr)):
            if np.array_equal(initial,arr[k]):
                frequency+=1
        frequencies.append(frequency/samples)
    return [states,frequencies]
     


def compute_expectation_pauli_zz(beta,A,b,samples,burn_in,i,j):
    empirical_average=0
    for k in range(0,samples):
        new_string=gen_samples_metropolis(beta,A,b,1,burn_in)
        empirical_average+=pauli_zz(new_string[0],i,j)
    return empirical_average/samples



def compute_expectation_pauli_zz_repeat(beta,A,b,samples,burn_in,i,j):
    empirical_average=0
    new_string=gen_samples_metropolis_repeat(beta,A,b,samples,burn_in)
    
    for k in range(0,samples):
        
        empirical_average+=pauli_zz(new_string[k],i,j)
    return empirical_average/samples



def compute_expectation_pauli_z(beta,A,b,samples,burn_in,i):
    empirical_average=0
    for k in range(0,samples):
        new_string=gen_samples_metropolis(beta,A,b,1,burn_in)
        empirical_average+=pauli_z(new_string[0],i)
    return empirical_average/samples
  



def compute_expect(A,beta,samples,burn_in):
    n=A.shape[0]
    correlator_list=A.nonzero()
    number_correlators=len(correlator_list[0])
    b=np.zeros(n)
    correlations=csr_matrix((np.zeros(len(correlator_list[0])), (correlator_list[0], correlator_list[1])))
    for k in range(0,number_correlators):
        correlations[correlator_list[0][k],correlator_list[1][k]]=compute_expectation_pauli_zz(beta,A,b,samples,burn_in,correlator_list[0][k],correlator_list[1][k])
    return 0.5*(correlations+np.transpose(correlations))



def compute_expect_repeat(A,beta,samples,burn_in):
    n=A.shape[0]
    correlator_list=A.nonzero()
    number_correlators=len(correlator_list[0])
    b=np.zeros(n)
    correlations=csr_matrix((np.zeros(len(correlator_list[0])), (correlator_list[0], correlator_list[1])))
    for k in range(0,number_correlators):
        print(k)
        correlations[correlator_list[0][k],correlator_list[1][k]]=compute_expectation_pauli_zz_repeat(beta,A,b,samples,burn_in,correlator_list[0][k],correlator_list[1][k])
    return 0.5*(correlations+np.transpose(correlations))





def compute_expect_given_target(C,A,beta,samples,burn_in):
    n=A.shape[0]
    correlator_list=A.nonzero()
    number_correlators=len(correlator_list[0])
    b=np.zeros(n)
    correlations=csr_matrix((np.zeros(len(correlator_list[0])), (correlator_list[0], correlator_list[1])))
    for k in range(0,number_correlators):
        print(k,burn_in)
        correlations[correlator_list[0][k],correlator_list[1][k]]=compute_expectation_pauli_zz(beta,C,b,samples,burn_in,correlator_list[0][k],correlator_list[1][k])
    return 0.5*(correlations+np.transpose(correlations))

def compute_expect_given_target_repeat(C,A,beta,samples,burn_in):
    n=A.shape[0]
    correlator_list=A.nonzero()
    number_correlators=len(correlator_list[0])
    b=np.zeros(n)
    correlations=csr_matrix((np.zeros(len(correlator_list[0])), (correlator_list[0], correlator_list[1])))
    for k in range(0,number_correlators):
        print(k,burn_in)
        correlations[correlator_list[0][k],correlator_list[1][k]]=compute_expectation_pauli_zz_repeat(beta,C,b,samples,burn_in,correlator_list[0][k],correlator_list[1][k])
    return 0.5*(correlations+np.transpose(correlations))



#compute_expectation_pauli_zz_repeat(0.1,A,b,10,30,0,1)

#sanity check repeated
#n=2
#beta=0.1
#b=np.zeros(n)
#A=generate_line(n)
#
#c=compute_expectation_pauli_zz(0.2,A,b,1000,20,0,1)
#c2=compute_expectation_pauli_zz(beta,A,b,2000,20,0,1)
#print(c,c1)
##sanity check tomography
#n=30
#A=generate_line_random(n)
#print(A.todense())
#iterations=10
#results=np.zeros([iterations,5])
#B=compute_expect(A,0.1,1000,100)
#correlator_list=A.nonzero()
#C=csr_matrix((np.zeros(len(correlator_list[0])), (correlator_list[0], correlator_list[1])))
#
#for k in range(0,iterations):
#    new_cov=compute_expect_given_target(C,A,0.1,1000,k*n*10)
#    C+=(new_cov-B)
#    D=(new_cov-B)
#    D=D.dot(C-A)
#    results[k,0]=k
#    results[k,1]=-np.trace(D.todense())
#    print(k,np.trace(D.todense()))
#    results_df=pd.DataFrame(results)
#    results_df.to_csv("tomography_gibbs.csv")  
#
#
#print(C)



#sanity test to see if external field sampling doing the correct thing
#n=6
#current=np.ones(n)
#
#C=0.000001*random_graph(n,1)
#
#b=np.ones(n)
#
#
#beta_test=0.3
#c=compute_expectation_pauli_z(beta_test,C,b,4000,100,2)
#
#print(c,(-np.exp(-beta_test)+np.exp(beta_test))/(np.exp(-beta_test)+np.exp(beta_test)))

#sanity check ZZ is working

#n=6
##current=np.ones(n)
##
#C=0.000001*random_graph(n,1)
#C[0,1]=1
#C[1,0]=1
#C[2,3]=1
#C[3,2]=1
##C[4,5]=1
##C[5,4]=1
#b=0.00001*np.ones(n)
#
#
#beta_test=0.1
##c=compute_expectation_pauli_zz(beta_test,C,b,5000,30,0,1)
#
#print(c,-(-np.exp(-beta_test)+np.exp(beta_test))/(np.exp(-beta_test)+np.exp(beta_test)))


#sanity test to see if partition function is working

#n=6
#current=np.ones(n)
#
#C=0.000001*random_graph(n,1)
#
#b=np.ones(n)
#
#
#beta_test=0.3
#schedule=[0,0.1,0.2]
#c=telescopic_product_external(C,b,schedule,2000,30)
#print(c,get_partitions_from_estimates(c),((np.exp(0.2)+np.exp(-0.2))/2)**n)





#strings=gen_samples_metropolis(0.1,C,b,10,400)
#a=compute_expectation_pauli_zz(0.1,C,b,100,400,2,3)

#
#n=100
#current=-np.ones(n)
#C=random_graph(n,2)
#x=random.randint(0,n)   
#blob=C.dot(current)
#energy=current.dot(blob)
#print("first energy",energy)
#B=C[x,:].nonzero()
#print(x,B[1])
#sum_neigh=0
#current[B[1][0]]=-1
#for k in range(0,len(B[1])):
#    sum_neigh+=C[x,B[1][k]]*current[B[1][k]]
#print(sum_neigh)
#for k in range(0,5):
#    [current,energy]=update(current,0.1,C,energy)
#    print(energy)



#
#n=11
#current=np.ones(n)
#
#C=random_graph(n,4)
#part=partition(C.todense(),0.15)
#
#b=0*-np.ones(n)
#
#schedule=np.linspace(0,0.15,3*n)
#estimates=telescopic_product_external(C,b,schedule,100,100)
#final=get_partitions_from_estimates(estimates)
#
#print(final[-1]/(n*np.log(2)),part)
#x=random.randint(0,n-1)   
#print(C.shape,current.shape)
#blob=C.dot(current)
#energy=current.dot(blob)
#print("first energy",energy)
#B=C[x,:].nonzero()
#print(x,B[1])
#sum_neigh=0
#current[B[1][0]]=-1
#for k in range(0,len(B[1])):
#    sum_neigh+=C[x,B[1][k]]*current[B[1][k]]
#print(sum_neigh)
#energies=[]
#for k in range(0,2000):
#    [current,energy]=update_external(current,0.4,C,b,energy)
#    
#    energies.append(energy)
#plt.plot(range(0,2000),energies)


#
#current=np.ones(n)
#x=random.randint(0,n-1)   
#print(C.shape,current.shape)
#blob=C.dot(current)
#energy=current.dot(blob)
#print("first energy",energy)
#B=C[x,:].nonzero()
#print(x,B[1])
#sum_neigh=0
#current[B[1][0]]=-1
#for k in range(0,len(B[1])):
#    sum_neigh+=C[x,B[1][k]]*current[B[1][k]]
#print(sum_neigh)
#energies=[]
#for k in range(0,2000):
#    [current,energy]=update_external(current,0.6,C,b,energy)
#    
#    energies.append(energy)
#plt.plot(range(0,2000),energies)




