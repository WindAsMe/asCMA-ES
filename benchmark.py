import copy

import numpy as np
########################################################################
#        _______  ___________   ______    ______   ___   ___
#       /  _____||___    ____| /  __  \  |   _  \  \  \ /  /
#      |  |  __      |  |     |  |  |  | |  |_)  |  \  V  /
#      |  | |_ |     |  |     |  |  |  | |   ___/    >   <
#      |  |__| |     |  |     |  |__|  | |  |       /  _  \
#       \______|     |__|      \______/  |__|      /__/ \__\
#
#                                                 Version 1.0
#       GTOPX - Space Mission Benchmarks
#       --------------------------------
#       This is an example program to test evaluate the ten
#       benchmark instances of GTOPX, which are:
#
#          No. 1  :   Cassini1
#          No. 2  :   Cassini2
#          No. 3  :   Messenger (reduced)
#          No. 4  :   Messenger (full)
#          No. 5  :   GTOC1
#          No. 6  :   Rosetta
#          No. 7  :   Sagas
#          No. 8  :   Cassini1-MINLP
#          No. 9  :   Cassini1-MO
#          No. 10 :   Cassini1-MO-MINLP
#
#       For each benchmark, the number of objetives (o), variables (n)
#       and constraints (m) are given. Note that benchmark 8 and 10 include
#       integer (discrete) variables, which are located at the end of the
#       solution vector "x". The arrays "xl" and "xu" denote the lower and
#       upper bounds (also called box-constraints) for each benchmark. The
#       array "x" given in this file is the best known solution vector.
#
#       For further information on GTOPX, see here:
#
#        http://www.midaco-solver.com/index.php/about/benchmarks/gtopx
#
#       For further information on ESA's original GTOP, see here:
#
#        https://www.esa.int/gsp/ACT/projects/gtop/
#
########################################################################
import ctypes
from ctypes import *
import os.path
########################################################################


pi = 3.14159265359


class GTOPX_Problem:
    def __init__(self, func_num):
        self.func_num = func_num
        if func_num == 1:
            self.o = 1  # number of objectives
            self.n = 6  # number of variables
            self.ni = 0  # number of integer variables
            self.m = 4  # number of constraints
            self.xl = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0]  # lower bounds
            self.xu = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0]  # upper bounds
        elif func_num == 2:
            self.o = 1  # number of objectives
            self.n = 22  # number of variables
            self.ni = 0  # number of integer variables
            self.m = 0  # number of constraints
            self.xl = [-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01, 1.05,
                       1.05, 1.15, 1.7, -pi, -pi, -pi, -pi]
            self.xu = [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0, 6.5,
                       291.0, pi, pi, pi, pi]
        elif func_num == 3:
            self.o = 1  # number of objectives
            self.n = 18  # number of variables
            self.ni = 0  # number of integer variables
            self.m = 0  # number of constraints
            self.xl = [1000.0, 1.0, 0.0, 0.0, 30.0, 30.0, 30.0, 30.0, 0.01, 0.01, 0.01, 0.01, 1.1, 1.1, 1.1, -pi, -pi,
                       -pi]
            self.xu = [4000.0, 5.0, 1.0, 1.0, 400.0, 400.0, 400.0, 400.0, 0.99, 0.99, 0.99, 0.99, 6.0, 6.0, 6.0, pi, pi,
                       pi]
        elif func_num == 4:
            self.o = 1  # number of objectives
            self.n = 26  # number of variables
            self.ni = 0  # number of integer variables
            self.m = 0  # number of constraints
            self.xl = [1900.0, 2.5, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.01, 0.01, 0.01, 0.01, 0.01,
                       0.01, 1.1, 1.1, 1.05, 1.05, 1.05, -pi, -pi, -pi, -pi, -pi]
            self.xu = [2300.0, 4.05, 1.0, 1.0, 500.0, 500.0, 500.0, 500.0, 500.0, 600.0, 0.99, 0.99, 0.99, 0.99, 0.99,
                       0.99, 6.0, 6.0, 6.0, 6.0, 6.0, pi, pi, pi, pi, pi]
        elif func_num == 5:
            self.o = 1  # number of objectives
            self.n = 8  # number of variables
            self.ni = 0  # number of integer variables
            self.m = 6  # number of constraints
            self.xl = [3000.0, 14.0, 14.0, 14.0, 14.0, 100.0, 366.0, 300.0]
            self.xu = [10000.0, 2000.0, 2000.0, 2000.0, 2000.0, 9000.0, 9000.0, 9000.0]
        elif func_num == 6:
            self.o = 1  # number of objectives
            self.n = 22  # number of variables
            self.ni = 0  # number of integer variables
            self.m = 0  # number of constraints
            self.xl = [1460.0, 3.0, 0.0, 0.0, 300.0, 150.0, 150.0, 300.0, 700.0, 0.01, 0.01, 0.01, 0.01, 0.01, 1.06,
                       1.05, 1.05, 1.05, -pi, -pi, -pi, -pi]
            self.xu = [1825.0, 5.0, 1.0, 1.0, 500.0, 800.0, 800.0, 800.0, 1850.0, 0.9, 0.9, 0.9, 0.9, 0.9, 9.0, 9.0,
                       9.0, 9.0, pi, pi, pi, pi]
        elif func_num == 7:
            self.o = 1  # number of objectives
            self.n = 12  # number of variables
            self.ni = 0  # number of integer variables
            self.m = 2  # number of constraints
            self.xl = [7000.0, 0.0, 0.0, 0.0, 50.0, 300.0, 0.01, 0.01, 1.05, 8.0, -pi, -pi]
            self.xu = [9100.0, 7.0, 1.0, 1.0, 2000.0, 2000.0, 0.9,  0.9, 7.0, 500.0, pi, pi]
        elif func_num == 8:
            self.o = 1  # number of objectives
            self.n = 10  # number of variables
            self.ni = 4  # number of integer variables
            self.m = 4  # number of constraints
            self.xl = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0, 1.0, 1.0, 1.0, 1.0]
            self.xu = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0, 9.0, 9.0, 9.0, 9.0]
        elif func_num == 9:
            self.o = 2  # number of objectives
            self.n = 6  # number of variables
            self.ni = 0  # number of integer variables
            self.m = 5  # number of constraints
            self.xl = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0]
            self.xu = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0]
        elif func_num == 10:
            self.o = 2  # number of objectives
            self.n = 10  # number of variables
            self.ni = 4  # number of integer variables
            self.m = 5  # number of constraints
            self.xl = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0, 1.0, 1.0, 1.0, 1.0]
            self.xu = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0, 9.0, 9.0, 9.0, 9.0]
        else:
            print("ERROR function number")

    def evaluate(self, x):
        if os.name == "posix":
            lib_name = "gtopx.so"  # Linux//Mac/Cygwin
        else:
            lib_name = "gtopx.dll"  # Windows
        lib_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib_name
        CLIB = ctypes.CDLL(lib_path)
        f_ = (c_double * self.o)()
        benchmark_ = c_long(self.func_num)
        x_ = (c_double * self.n)()
        for i in range(0, self.n):
            x_[i] = c_double(x[i])
        if self.m > 0:
            g_ = (c_double * self.m)()
        if self.m == 0:
            g_ = (c_double * 1)()
        CLIB.gtopx(benchmark_, f_, g_, x_)
        f = [0.0] * self.o
        g = [0.0] * self.m
        for i in range(0, self.o):
            f[i] = f_[i]
        for i in range(0, self.m):
            g[i] = g_[i]
        return f, g

    def info(self):
        return self.o, self.n, self.ni, self.m, self.xl, self.xu


def base_fitness(Population, func, intercept):
    fitness = []
    for indi in Population:
        fitness.append(func(indi)[0][0] - intercept)
    return fitness


def group_individual(group, individual, center):
    part_individual = copy.deepcopy(center)
    for element in group:
        part_individual[element] = individual[element]
    return part_individual


def groups_fitness(groups, Population, func, cost, intercept, center):
    fitness = []
    for indi in Population:
        indi_fitness = 0
        for group in groups:
            indi_fitness += (func(group_individual(group, indi, center))[0][0] - intercept)
            cost += 1
        fitness.append(indi_fitness)
    return fitness, cost


# outer interface
# opt_fitness is calculated by groups_fitness
def object_function(base_fitness, opt_fitness):
    error = 0
    for i in range(len(base_fitness)):
        error += ((base_fitness[i] - opt_fitness[i])) ** 2
    return error

