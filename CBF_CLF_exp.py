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



def main():
    Params('configs.ini')
    np.random.seed(2)
    random.seed(2)
    sys.setrecursionlimit(2000)
    worldChar = 'TestingCol'
    if worldChar is 'Cltrd_world':
        q_init = np.array([1, 2, 0])
        xy_goal = np.array([5.5, 4.5])
        q_init = np.array([5, 5, 0])
        xy_goal = np.array([3, 6])
    elif worldChar is 'Cltrd_world_big':
        q_init = np.array([6.5, 5, 0])
        xy_goal = np.array([3, 5])
    elif worldChar is 'nPsgs_world':
        q_init = np.array([0, 0, 0])
        xy_goal = np.array([24., 16.])
    elif worldChar is 'Smpl_world':
        q_init = np.array([3., 4.1, 0])
        xy_goal = np.array([7, 4])
    elif worldChar is 'NoObs_world':
        q_init = np.array([0, 0, 0])
        xy_goal = np.array([20., 16.])
    elif worldChar is 'Circ_world':
        q_init = np.array([2, 13, 0])
        xy_goal = np.array([23, 12.5])
    elif worldChar is 'Cross1_world':
        q_init = np.array([-1,-.5, 0])
        xy_goal = np.array([2,2])
    elif worldChar is 'TestingCol':
        q_init = np.array([0.5,.5, 0])
        # xy_goal = np.array([1.75,-.25])
        # xy_goal = np.array([1.75,-.24])
        xy_goal = np.array([1.75,-.8])
    # File naming stuff:

    CBF_RRT_object = CBF_RRTstrr(suffix=worldChar, q_init=q_init, xy_goal=xy_goal, eps_g=.5)
    if CBF_RRT_object.params.kde_enabled or CBF_RRT_object.params.CE_enabled:
        prefix = 'adap'
    else:
        prefix = '_'
    CBF_RRT_object.prefix = prefix
    CBF_RRT_object.Initialization(worldChar=worldChar)
    CBF_RRT_object.initialize_graphPlot()
    # ---- Manual plan a single long edge:
    xy_v_new = q_init[0:2]
    mSteering2goal = np.linalg.norm(xy_goal - xy_v_new) / CBF_RRT_object.params.step_size

    theta_goal = math.atan2(xy_goal[1] - xy_v_new[1], xy_goal[0] - xy_v_new[0])
    # qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = CBF_RRT_object.SafeSteering(xy_v_new,
    #                                                                           desired_theta=theta_goal,
    #                                                                           m=int(1.5*mSteering2goal))
    qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = CBF_RRT_object.SafeSteering(xy_v_new,
                                                                              desired_pos=xy_goal)                                                                       
    coll_falg = CBF_RRT_object.iscoll(qTraj=qTrajectory)
    CBF_RRT_object.plot_pathToVertex(qTrajectory,EnPlotting=True)
    CBF_RRT_object.TreePlot.show()

    #Compute the control inputs in terms of \omega and v: 

    # uIn = omega_vCtrl(uIn=uTrajectory,xTraj=qTrajectory)

    saveData([qTrajectory,uIn], 'CDC_cbfclfLP_ctrl_traj_data', suffix=CBF_RRT_object.suffix, CBF_RRT_strr_obj=CBF_RRT_object,
                     adapDist_iter=CBF_RRT_object.adapIter-1, enFlag=False)
    
    fig1, ax1 = plt.subplots()
    uTrajectory = np.asanyarray(uTrajectory[0])
    u1_in = uTrajectory[:,0]
    u2_in = uTrajectory[:,1]
    ax1.plot(tTrajectory,u1_in)
    plt.show()
    a = 1

if __name__ == '__main__':

    main()