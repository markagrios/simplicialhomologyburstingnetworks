import numpy as np
import sys
import os
from mogutda import SimplicialComplex

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
        maxdim = 0
        for num in range(len(line)):
            line[num] = int(line[num])

        simps.append(tuple(line))

    return(simps)

#######################

matrix_name = sys.argv[1]

simplist_from_matrix(matrix_name)
simps = mogulist_from_simplist(matrix_name)
for i in range(len(simps)):
    print(simps[i])

SC = SimplicialComplex(simplices=simps)
bettis = [SC.betti_number(0),SC.betti_number(1),SC.betti_number(2),SC.betti_number(3),SC.betti_number(4),SC.betti_number(5),SC.betti_number(6)]
print(bettis)
