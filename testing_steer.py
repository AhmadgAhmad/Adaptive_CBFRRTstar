import sys
print(sys.version)
import time
import timeit
import numpy as np
import matplotlib.cm as cm
import os
from matplotlib.backends.backend_pdf import PdfPages
from dynamics import Dyn, SingleIntegrator
from agent import Agent
from obstacle import Sphere, Ellipsoid, Wall
from gurobipy import *
from goal import Goal
from sim_plots import Cbf_data, Clf_data, ColorManager
from params import *
# from hocbf_dummy import HOCBF
from hocbf import *
from simulation import Simulation
import copy
import random
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from vertex import Vertex
from expansionTree import ExpansionTree
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn import mixture
import sklearn
from scipy.stats import multivariate_normal
import pickle
from optimizingCBF_RRTstr import *
from steer import Steer

def main(testing_scen):
    Params('configs.ini')
    np.random.seed(2)
    random.seed(2)
    sys.setrecursionlimit(2000)
    worldChar = 'TestingCol'
    if worldChar is 'Cltrd_world':
        q_init = np.array([1, 2, 0],dtype='float32')
        q_goal = np.array([5.5, 4.5],dtype='float32')
        q_init = np.array([5, 5, 0],dtype='float32')
        q_goal = np.array([3, 6,0],dtype='float32')
    elif worldChar is 'Cltrd_world_big':
        q_init = np.array([6.5, 5, 0],dtype='float32')
        q_goal = np.array([3, 5,0],dtype='float32')
    elif worldChar is 'nPsgs_world':
        q_init = np.array([0, 0, 0],dtype='float32')
        q_goal = np.array([24., 16.,0],dtype='float32')
    elif worldChar is 'Smpl_world':
        q_init = np.array([3., 4.1, 0],dtype='float32')
        q_goal = np.array([7, 4,0],dtype='float32')
    elif worldChar is 'NoObs_world':
        q_init = np.array([0, 0, 0],dtype='float32')
        q_goal = np.array([20., 16.,0],dtype='float32')
    elif worldChar is 'Circ_world':
        q_init = np.array([2, 13, 0],dtype='float32')
        q_goal = np.array([23, 12.5,0],dtype='float32')
    elif worldChar is 'Cross1_world':
        q_init = np.array([-1,-.5, 0],dtype='float32')
        q_goal = np.array([2,2,0],dtype='float32')
    elif worldChar is 'TestingCol':
        q_init = np.array([-0.5,0, 0],dtype='float32')
        q_goal = np.array([2,-.51, 0],dtype='float32')
    # File naming stuff:

    CBF_RRT_object = CBF_RRTstrr(suffix=worldChar, q_init=q_init, xy_goal=q_goal[0:2], eps_g=.5)
    if CBF_RRT_object.params.kde_enabled or CBF_RRT_object.params.CE_enabled:
        prefix = 'adap'
    else:
        prefix = '_'
    CBF_RRT_object.prefix = prefix
    CBF_RRT_object.Initialization(worldChar=worldChar)
    CBF_RRT_object.initialize_graphPlot()
    # ---- Manual plan a single long edge:
    xy_v_new = q_init[0:2]
    mSteering2goal = np.linalg.norm(q_goal[0:2] - xy_v_new) / CBF_RRT_object.params.step_size

    theta_goal = math.atan2(q_goal[1] - xy_v_new[1], q_goal[0] - xy_v_new[0])
    qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = CBF_RRT_object.SafeSteering(xy_v_new,
                                                                              desired_pos=[q_goal[0],q_goal[1]],
                                                                              m=int(mSteering2goal))

    
    CBF_RRT_object.plot_pathToVertex(np.asarray(qTrajectory),EnPlotting=True)
    CBF_RRT_object.TreePlot.show()
    
    
    # Define a steer instance: 
    simObject = CBF_RRT_object.simObject  # TODO (meeting) create a reset option,
    
    steer_obj = Steer(q_i = q_init, q_f = q_goal, simObject = simObject)
    qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = steer_obj.safeSteer(q_i = q_init, q_f = q_goal)

    #Choose the testing scenario (a single trajectoy/ multiple ones with concanation):
    if testing_scen is 'single_traj':
        pass
    elif testing_scen is 'multi_traj': 
        pass

if __name__ == '__main__':
    testing_scen = 'single_traj'
    # testing_scen = 'multi_traj'
    main(testing_scen)