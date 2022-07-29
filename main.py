import numpy as np
from util import helps
from Optimizer import DE, ES, GA, CMAES
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
    # Evaluation times = Dim * 15000
    # NIND = Dim * 50
    this_path = path.realpath(__file__)
    trial_run = 30

    for func_num in [2]:

        """Parameter initialization"""
        problem = benchmark.GTOPX_Problem(func_num)
        obj, var, int_var, cons, LB, UB = problem.info()
        func = problem.evaluate
        Dim = var
        int_var_start = Dim - int_var

        NIND = Dim * 100
        FEs = Dim * 10000
        scale_range = [LB, UB]
        Max_iter = int(FEs / NIND)
        VarTypes = [0] * Dim
        for i in range(int_var_start, Dim):
            VarTypes[i] = 1

        """Path definition"""
        DE_obj_path = path.dirname(this_path) + '/data/obj/DE/obj/f' + str(func_num)
        DE_CV_path = path.dirname(this_path) + '/data/obj/DE/cv/f' + str(func_num)
        ES_obj_path = path.dirname(this_path) + '/data/obj/ES/obj/f' + str(func_num)
        ES_CV_path = path.dirname(this_path) + '/data/obj/ES/cv/f' + str(func_num)
        GA_obj_path = path.dirname(this_path) + '/data/obj/GA/obj/f' + str(func_num)
        GA_CV_path = path.dirname(this_path) + '/data/obj/GA/cv/f' + str(func_num)

        """Trial run"""
        for i in range(trial_run):

            """Optimization with DE"""
            DE_obj_trace, DE_best_indi_CV = DE.DE_exe(Dim, Max_iter, NIND, problem.evaluate, scale_range, VarTypes)
            write_obj(DE_obj_trace, DE_obj_path)
            write_obj(helps.CV_Normalize(DE_best_indi_CV), DE_CV_path)

            """Optimization with ES"""
            ES_obj_trace, ES_best_indi_CV = ES.ES_exe(Dim, Max_iter, NIND, problem.evaluate, scale_range, VarTypes)
            write_obj(ES_obj_trace, ES_obj_path)
            write_obj(helps.CV_Normalize(ES_best_indi_CV), ES_CV_path)

            """Optimization with GA"""
            GA_obj_trace, GA_best_indi_CV = GA.GA_exe(Dim, Max_iter, NIND, problem.evaluate, scale_range, VarTypes)
            write_obj(GA_obj_trace, GA_obj_path)
            write_obj(helps.CV_Normalize(GA_best_indi_CV), GA_CV_path)




