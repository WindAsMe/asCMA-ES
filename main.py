import numpy as np
from util import helps
from DE import DE
import benchmark
import math
from os import path


def write_obj(data, path):
    with open(path, 'a') as f:
        f.write(str(data) + ', ')
        f.write('\n')
        f.close()


def Normal(Dim):
    group = []
    for i in range(Dim):
        group.append(i)
    return [group]


if __name__ == "__main__":
    # Target function is [1, 8]
    # Evaluation times = Dim * 3000
    # NIND = Dim * 50
    this_path = path.realpath(__file__)
    trial_run = 30

    for func_num in [2]:

        """Parameter initialization"""
        problem = benchmark.GTOPX_Problem(func_num)
        obj, var, int_var, cons, LB, UB = problem.info()
        func = problem.evaluate
        Dim = var
        Gene_len = int(math.log2(Dim)) - 1
        pop_size = 5
        epsilon = 0.001
        max_len = int(var / 2)
        int_var_start = Dim - int_var
        print(int_var_start)
        NIND = Dim * 30
        FEs = Dim * 1500
        scale_range = [LB, UB]
        for i in range(trial_run):

            """Trial run"""

            """Normal method"""
            Normal_Max_iter = int(FEs / NIND)
            Normal_Vars = [0] * Dim
            for i in range(int_var_start, Dim):
                Normal_Vars[i] = 1
            Normal_obj_trace, Normal_best_indi_CV = DE.DE_exe(Dim, Normal_Max_iter, NIND, problem.evaluate, scale_range,
                                                                       Normal_Vars)
            Normal_obj_path = path.dirname(this_path) + '/Data/obj/DE/f' + str(func_num)
            Normal_CV_path = path.dirname(this_path) + '/Data/CV/DE/f' + str(func_num)
            write_obj(Normal_obj_trace, Normal_obj_path)
            write_obj(helps.CV_Normalize(Normal_best_indi_CV), Normal_CV_path)


