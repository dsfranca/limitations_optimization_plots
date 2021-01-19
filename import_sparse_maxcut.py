#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:29:05 2020

@author: danielstilckfranca
"""

from scipy.sparse import csr_matrix, find
import random
import numpy as np
import time
import itertools
import pandas as pd

def import_instance(name_file):
    
    
    
    file1 = open(name_file, 'r') 
    Lines = file1.readlines()
    line0=Lines[0].split()
    n=int(line0[0])
    m=len(Lines)
    rows=[]
    columns=[]
    entries=[]
    for k in range(1,m):
        info_entry=Lines[k].split()
        rows.append(int(info_entry[0])-1)
        rows.append(int(info_entry[1])-1)
        columns.append(int(info_entry[1])-1)
        columns.append(int(info_entry[0])-1)
        entries.append(0.5*int(info_entry[2]))
        entries.append(0.5*int(info_entry[2]))
    A=csr_matrix((entries, (rows, columns)))
    return A
        
        
    







