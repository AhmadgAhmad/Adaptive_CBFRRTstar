from cmath import pi
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


class CE_GMM(object):
    def __init__(self,sNoise,mus,covs):
        self.k = 0
        self.mus = mus
        self.covs = covs
        self.sNoise = sNoise

    def gmmSample(self):
        k = len(self.covs)
        mus = self.mus
        covs = self.covs
        #With equal probability choose one of the mixing distributions to sample from:
        if k > 1:
            ik = np.random.randint(0,k-1)
        else:
            ik = 0
        xySample = np.random.multivariate_normal(mean=mus[ik],cov=covs[ik]+self.sNoise*np.identity(2),size=1)

        return xySample[0]

######################################################################################################################:
############################## The main body of Algorithm 1:##########################################################:
class CBF_RRTstrr(object):
    def __init__(self, suffix,q_init=np.array([0, 0, 0]), xy_goal=np.array([15,15]), eps_g=0.1, NSampling=500, AdSamplingFlag=False,
                 TreeT=None):
        self.params = Params()
        self.q_init = q_init
        self.xy_goal = xy_goal
        self.eps_g = eps_g
        self.NSampling = NSampling
        self.th = self.params.step_size
        self.AdSamplingFlag = AdSamplingFlag
        self.TreeT = TreeT
        self.TreePlot = None
        self.TrajPlot = None
        self.Vg_leaves = []
        self.CBF_RRTstrEnable = self.params.CBF_RRTstrEnable
        self.initEliteSamples = []
        self.goalReached = False
        self.suffix = suffix
        self.prefix = None

        #Previous and current addaptation parama
        self.adapIter = 1
        self.kde_preSamples = []
        self.kde_currSamples =[]
        self.curr_Ldist = 0
        self.prev_Ldist = 0
        #Reaching the optimal distribution params:
        #---kde
        self.kdeOpt_flag = False
        self.kde_eliteSamples = []
        self.KDE_fitSamples = None
        self.KDE_pre_gridProbs = None
        self.kde_enabled = self.params.kde_enabled
        #---gmm
        self.gmmOpt_flag = False

        #Sampling att
        self.ceGMM = None
        self.pre_mus = None
        self.opt_mus = None
        self.opt_covs = None

        #initial tree attributes
        self.initTree_done = False

        #Eliette samples and CE comutation att
        self.len_frakX = 0
        self.pre_gridProbs = []

        #Attributes to save data to use it afterwards:
        self.goal_costToCome_list = None
        self.vg_minCostToCome_list = None
        self.i = None
        self.iGoalReached = None
        self.iterTime_list=None
        self.iRun = None
        
        # Attributes to count samples, accepted ones and rejected ones. 
        self.smpl_tr = 0
        self.smpl_axpd = 0 

        #simulation Attributes:
        self.obsWorldList = []
        self.simObject = None
        self.robotDynType = Dyn.SINGLE_INT
        
        #Probing attributes:
        self.bestGCost = None
        self.Vexisit = False
        
        #CBF-RRT: 
        self.CBF_RRT_enable = self.params.CBF_RRT_enabled

        # RRTstar: 
        self.RRTstr_enabled = self.params.RRTstr_enabled 

        #CLF-CBF QP: 
        self.CBF_CLF_enabled = self.params.CBF_CLF_enabled
    ######################################################################################################################:
    ############################## Initialization Procedure: #############################################################:
     # This method initializes the tree with the an initial vertex and empty edge set. It also initiates a
     # simulation object that contains the obstacles in the mission space.
    def Initialization(self,worldChar):

        #The initilaization paramaters:
        initTree_flag = self.params.initTree_flag
        q_init = self.q_init
        v_init = Vertex(State=q_init, PathToCome=[0],indexID=0)
        Vg_leaves = None
        TreeT = ExpansionTree()  # An expansion tree class.
        TreeT.addVertex(v_init)
        v_init.computeCostToCome(TreeT=TreeT)
        self.TreeT = TreeT
        self.Vg_leaves = []
        if worldChar is 'Cltrd_world':
            obs1 = Ellipsoid(np.array([3,3]), np.array([.6, .1]), angle=-45)
            obs2 = Ellipsoid(np.array([3,2.5]), np.array([.6, .1]), angle=30)
            obs3 = Ellipsoid(np.array([4,4]), np.array([.6, .1]), angle=-60)
            obs4 = Ellipsoid(np.array([4, 4.5]), np.array([.6, .1]), angle=60)
            obs5 = Ellipsoid(np.array([6,3]), np.array([.6, .1]), angle=-80)
            obs6 = Ellipsoid(np.array([4,6]), np.array([.6, .1]), angle=-45)
            # obs6  = Sphere([8,3], radius=.5)
            # obs7 = Sphere([2, 4.5], radius=.5)

            obs7 = Ellipsoid(np.array([4.5, 5]), np.array([.6, .1]), angle=45)
            # obs8 = Ellipsoid([], [.6, .05], angle=0)
            # obs9 = Ellipsoid([], [.6, .05], angle=-45)
            # #Cross:
            # obs10 = Ellipsoid([], [.6, .05], angle=45)
            # obs11 = Ellipsoid([], [.6, .05], angle=-45)

            self.add_obstacleToWorld(obs1)
            self.add_obstacleToWorld(obs2)
            self.add_obstacleToWorld(obs3)
            self.add_obstacleToWorld(obs4)
            self.add_obstacleToWorld(obs5)
            #
            self.add_obstacleToWorld(obs6)
            self.add_obstacleToWorld(obs7)
            # self.add_obstacleToWorld(obs8)
            # self.add_obstacleToWorld(obs9)
            # self.add_obstacleToWorld(obs10)
            # self.add_obstacleToWorld(obs11)

        if worldChar is 'Cltrd_world_big':
            # obs1 = Ellipsoid([3, 3.8], [1.1, .45], angle=-45)
            # obs2 = Ellipsoid([3, 3], [1, .2], angle=30)
            # obs3 = Ellipsoid([5, 4.7], [.8, .2], angle=45)
            # obs4 = Ellipsoid([6, 7.6], [.8, .2], angle=45)
            # obs5 = Ellipsoid([10, 3], [.8, .2], angle=-80)
            # obs6 = Ellipsoid([2, 7.5], [.8, .2], angle=45)
            # # obs6  = Sphere([8,3], radius=.5)
            # # obs7 = Sphere([2, 4.5], radius=.5)
            #
            # obs7 = Ellipsoid([8.5, 5], [.8, .2], angle=45)
            # obs8 = Ellipsoid([6.5, 1], [.6, .6], angle=-20)
            # # #Cross:
            # obs10 = Ellipsoid([], [.6, .05], angle=45)
            # obs11 = Ellipsoid([], [.6, .05], angle=-45)
            obs1 = Ellipsoid(np.array([4, 5]),np.array([1, 1]), angle=-45)
            obs2 = Ellipsoid(np.array([6, 0]), np.array([1, 1]), angle=30)
            obs3 = Ellipsoid(np.array([5, 4.7]),np.array( [1, 1]), angle=45)
            obs4 = Ellipsoid(np.array([6, 7.6]), np.array([1,1]), angle=45)
            obs5 = Ellipsoid(np.array([10, 3]), np.array([1,1]), angle=-80)
            obs6 = Ellipsoid(np.array([2, 7.5]), np.array([1,1]), angle=45)
            # obs6  = Sphere([8,3], radius=.5)
            # obs7 = Sphere([2, 4.5], radius=.5)

            # obs7 = Ellipsoid([8.5, 5], [.8, .8], angle=45)
            # obs8 = Ellipsoid([6.5, 1], [.6, .6], angle=-20)
            # # #Cross:
            # obs10 = Ellipsoid([], [.6, .6], angle=45)
            # obs11 = Ellipsoid([], [.6, .6], angle=-45)

            self.add_obstacleToWorld(obs1)
            self.add_obstacleToWorld(obs2)
            self.add_obstacleToWorld(obs3)
            self.add_obstacleToWorld(obs4)
            self.add_obstacleToWorld(obs5)
            # # #
            # self.add_obstacleToWorld(obs6)
            # self.add_obstacleToWorld(obs7)
            # self.add_obstacleToWorld(obs8)

        if worldChar is 'Cltrd_world_big2':
            # obs1 = Ellipsoid([3, 4], [1.1, .2], angle=90)
            # obs2 = Ellipsoid([3, .15], [.8, .2], angle=90)
            # obs3 = Ellipsoid([3, 7.85], [.8, .2], angle=90)
            #
            # obs4 = Ellipsoid([6, 2], [.8, .2], angle=90)
            # obs5 = Ellipsoid([6, 6], [.8, .2], angle=90)
            #
            # obs6 = Ellipsoid([8, 8], [1.1, .2], angle=90)
            # obs7 = Ellipsoid([8, 0], [1.1, .2], angle=90)
            #
            # # #Cross:
            # obs10 = Ellipsoid([], [.6, .05], angle=45)
            # obs11 = Ellipsoid([], [.6, .05], angle=-45)
            obs1 = Ellipsoid([3, 4], [1.1, 1.1], angle=90)
            obs2 = Ellipsoid([3, .15], [.8, .8], angle=90)
            obs3 = Ellipsoid([3, 7.85], [.8, .8], angle=90)

            obs4 = Ellipsoid([6, 2], [.8, .8], angle=90)
            obs5 = Ellipsoid([6, 6], [.8, .8], angle=90)

            obs6 = Ellipsoid([8, 8], [1.1, 1.1], angle=90)
            obs7 = Ellipsoid([8, 0], [1.1, 1.1], angle=90)

            # #Cross:
            obs10 = Ellipsoid([], [.6, .6], angle=45)
            obs11 = Ellipsoid([], [.6, .6], angle=-45)
            self.add_obstacleToWorld(obs1)
            self.add_obstacleToWorld(obs2)

            self.add_obstacleToWorld(obs3)
            self.add_obstacleToWorld(obs4)
            self.add_obstacleToWorld(obs5)

            self.add_obstacleToWorld(obs6)
            self.add_obstacleToWorld(obs7)
            # self.add_obstacleToWorld(obs8)
        if worldChar is 'nPsgs_world':
            obs1 = Ellipsoid([2.9, 3], [1.1, .3], angle=90)
            obs2 = Ellipsoid([4.5, 0], [1, .2], angle=90)
            obs3 = Ellipsoid([6, 4.8], [.8, .2], angle=90)
            obs4 = Ellipsoid([5, 8], [.8, .2], angle=90)
            obs5 = Ellipsoid([10, 3], [.8, .2], angle=90)
            obs6 = Ellipsoid([2, 7.8], [.8, .2], angle=90)
            # obs6  = Sphere([8,3], radius=.5)
            # obs7 = Sphere([2, 4.5], radius=.5)

            obs7 = Ellipsoid([9, 4.8], [.8, .2], angle=90)
            obs8 = Ellipsoid([3.5, 6.2], [.6, .1], angle=45)
            # #Cross:
            obs10 = Ellipsoid([], [.6, .05], angle=45)
            obs11 = Ellipsoid([], [.6, .05], angle=-45)

            self.add_obstacleToWorld(obs1)
            self.add_obstacleToWorld(obs2)
            self.add_obstacleToWorld(obs3)
            self.add_obstacleToWorld(obs4)
            self.add_obstacleToWorld(obs5)
            # #
            self.add_obstacleToWorld(obs6)
            self.add_obstacleToWorld(obs7)
            self.add_obstacleToWorld(obs8)

        if worldChar is 'Smpl_world':
            obs1 = Ellipsoid([5, 4], [.4, .21], angle=45)
            obs2 = Ellipsoid([5, 4], [.4, .21], angle=-45)

            self.add_obstacleToWorld(obs1)
            # self.add_obstacleToWorld(obs2)

        if worldChar is 'NoObs_world':
            pass

        if worldChar is 'TestingCol':
            obs1 = Sphere(np.array([10,10]), radius=5)
            obs2 = Ellipsoid(np.array([2,0]), np.array([.15,.5]), angle=45)
            obs3 = Ellipsoid(np.array([2, 0]),np.array([.15,0.5]), angle=-45)
            #--- self.add_obstacleToWorld(obs1)
            self.add_obstacleToWorld(obs2)
            self.add_obstacleToWorld(obs3)

        if worldChar is 'Circ_world':
            #This world to demonstrad a carfully designed world where multiple paths with have identical cost
            r = .5
            obs1 = Sphere([8, 24.5], radius=r)
            obs2 = Sphere([8, 18], radius=r)
            obs3 = Sphere([8, 12.5], radius=r)
            obs4 = Sphere([8, 6.5], radius=r)
            obs5 = Sphere([8, .5], radius=r)

            obs6 = Sphere([13, 21.4], radius=r)
            obs7 = Sphere([13, 15.8], radius=r)
            obs8 = Sphere([13, 9.2], radius=r)
            obs9 = Sphere([13, 3.6], radius=r)

            obs10 = Sphere([18, 24.5], radius=r)
            obs11 = Sphere([18, 18.5], radius=r)
            obs12 = Sphere([18, 12.5], radius=r)
            obs13 = Sphere([18, 6.5], radius=r)
            obs14 = Sphere([18, .5], radius=r)


            self.add_obstacleToWorld(obs1)
            self.add_obstacleToWorld(obs2)
            self.add_obstacleToWorld(obs3)
            self.add_obstacleToWorld(obs4)
            self.add_obstacleToWorld(obs5)
            self.add_obstacleToWorld(obs6)
            self.add_obstacleToWorld(obs7)
            self.add_obstacleToWorld(obs8)
            self.add_obstacleToWorld(obs9)
            self.add_obstacleToWorld(obs10)
            self.add_obstacleToWorld(obs11)
            self.add_obstacleToWorld(obs12)
            self.add_obstacleToWorld(obs13)
            self.add_obstacleToWorld(obs14)
        if worldChar is 'RSS_paper': 
            # This world to demonstrad a carfully designed world where multiple paths with have identical cost

            obs1 = Ellipsoid(np.array([9, 3]),np.array([1.5,1.5]),angle=0)
            obs2 = Ellipsoid(np.array([0, 8.5]), np.array([1.5, 1.5]), angle=0)
            obs3 = Ellipsoid(np.array([9, 16.5]), np.array([1.5, 1.5]), angle=0)
            obs4 = Ellipsoid(np.array([15, 5]), np.array([1.5, 1.5]), angle=0)
            obs5 = Ellipsoid(np.array([5, 6]), np.array([1.1, 1.1]), angle=0)

            # obs6 = Ellipsoid(np.array([13, 19.4]), np.array([1.5, 1.5]), angle=0)
            obs7 = Ellipsoid(np.array([10, 10.5]), np.array([1.8, .4]), angle=60)
            obs8 = Ellipsoid(np.array([10, 10.5]), np.array([1.8, .4]), angle=-30)

            # obs9 = Ellipsoid(np.array([13, 5.6]), np.array([1.5, 1.5]), angle=0)
            # obs10 = Ellipsoid(np.array([17, 22.5]), np.array([1.5, 1.5]), angle=0)
            # obs11= Ellipsoid(np.array([17, 17.5]),np.array([1.5, 1.5]), angle=0)
            # obs13 = Ellipsoid(np.array([17, 7.5]), np.array([1.5, 1.5]), angle=0)
            # obs14= Ellipsoid(np.array([17, 2.5]), np.array([1.5, 1.5]), angle=0)

            self.add_obstacleToWorld(obs1)
            self.add_obstacleToWorld(obs2)
            self.add_obstacleToWorld(obs3)
            self.add_obstacleToWorld(obs4)
            self.add_obstacleToWorld(obs5)
            # self.add_obstacleToWorld(obs6)
            self.add_obstacleToWorld(obs7)
            self.add_obstacleToWorld(obs8)
            # self.add_obstacleToWorld(obs9)
            # self.add_obstacleToWorld(obs10)
            # self.add_obstacleToWorld(obs11)
            # # self.add_obstacleToWorld(obs12)
            # self.add_obstacleToWorld(obs13)
            # self.add_obstacleToWorld(obs14)
        if worldChar is 'CircCross_world':
            # This world to demonstrad a carfully designed world where multiple paths with have identical cost

            obs1 = Ellipsoid(np.array([9, 21.5]),np.array([1.5,1.5]),angle=0)
            obs2 = Ellipsoid(np.array([9, 17]), np.array([1.5, 1.5]), angle=0)
            obs3 = Ellipsoid(np.array([9, 12.5]), np.array([1.5, 1.5]), angle=0)
            obs4 = Ellipsoid(np.array([9, 8]), np.array([1.5, 1.5]), angle=0)
            obs5 = Ellipsoid(np.array([9, 3.5]), np.array([1.5, 1.5]), angle=0)

            obs6 = Ellipsoid(np.array([13, 19.4]), np.array([1.5, 1.5]), angle=0)
            obs7 = Ellipsoid(np.array([14.8, 11.4]), np.array([2, .4]), angle=45)
            obs8 = Ellipsoid(np.array([14.8, 13.2]), np.array([2, .4]), angle=-45)

            obs9 = Ellipsoid(np.array([13, 5.6]), np.array([1.5, 1.5]), angle=0)
            obs10 = Ellipsoid(np.array([17, 22.5]), np.array([1.5, 1.5]), angle=0)
            obs11= Ellipsoid(np.array([17, 17.5]),np.array([1.5, 1.5]), angle=0)
            obs13 = Ellipsoid(np.array([17, 7.5]), np.array([1.5, 1.5]), angle=0)
            obs14= Ellipsoid(np.array([17, 2.5]), np.array([1.5, 1.5]), angle=0)

            self.add_obstacleToWorld(obs1)
            self.add_obstacleToWorld(obs2)
            self.add_obstacleToWorld(obs3)
            self.add_obstacleToWorld(obs4)
            self.add_obstacleToWorld(obs5)
            self.add_obstacleToWorld(obs6)
            self.add_obstacleToWorld(obs7)
            self.add_obstacleToWorld(obs8)
            self.add_obstacleToWorld(obs9)
            self.add_obstacleToWorld(obs10)
            self.add_obstacleToWorld(obs11)
            # self.add_obstacleToWorld(obs12)
            self.add_obstacleToWorld(obs13)
            self.add_obstacleToWorld(obs14)
        if worldChar is 'Cross1_world':
            # obs1 = Ellipsoid([8.8, 10.8], [2, .2], angle=45)
            # obs2 = Sphere([8.8, 12.5], radius=1.5)
            # obs2 = Ellipsoid([8.8, 13.5], [2, .51], angle=0)
            obs1 = Sphere([0.3, 1.2], radius=0.2)
            obs2 = Sphere([1, .5], radius=0.2)
            obs3 = Sphere([1.7, -.5], radius=0.2)
            obs4 = Ellipsoid([1.5, 1.5], [.4, .05], angle=45)
            obs5 = Ellipsoid([1.5, 1.5], [.4, .05], angle=90)
            obs6 = Ellipsoid([0, .5], [.4, .05], angle=0)
            obs7 = Ellipsoid([3, 1.6], [.4, .05], angle=90)

            # self.add_obstacleToWorld(obs1)
            # self.add_obstacleToWorld(obs2)
            # self.add_obstacleToWorld(obs3)
            self.add_obstacleToWorld(obs4)
            self.add_obstacleToWorld(obs5)
            self.add_obstacleToWorld(obs6)
            self.add_obstacleToWorld(obs7)

        self.simObject = Simulation()  # TODO (meeting) create a reset option,
        obsList = self.obsWorldList
        [self.simObject.add_obstacle(obs) for obs in obsList]

        return TreeT, Vg_leaves

    def add_obstacleToWorld(self,obs):
        """
        Append the obstacle to the obstacles list. The list is used when defining a simulation instance to safe steer.
        :param obs: obstacle simulation object.
        :return: -
        """
        if len(self.obsWorldList) == 0:
            self.obsWorldList = [obs]
        else:
            curr_obsWorldList = self.obsWorldList
            curr_obsWorldList.append(obs)
            self.obsWorldList = curr_obsWorldList

    ######################################################################################################################:
    ############################# Update Sampling Algorithm 5: #############################################################:
    def UpSampling_andSample(self, Vg_leaves,length,width,initTree_flag, rho = 0.1, rce = .5 ,md=8, k=4, n=2):
        """
        Generate samples from the 2D Euclidean space. Generate samples uniformly until enough trajectories which reach the goal;
        where th samples of these trajectories are described and used to optimize the sampling distribution. The sampling distribution
        will be adapted by the elite samples.

        :param Vg_leaves: (n*1) list of the vertexes of the tree that we succeeded to connect to the goal.
        :param length: the workspace dimension
        :param width:  the workspace dimension
        :param initTree_flag:
        :param rho: the probability of the rare event
        :param rce: the probability of exploring the workspace, where the complement probability is for exploiting the samples of the trajectories of Vg_leaves
        :param md: the number of samples of discretizing the trajectories
        :param k:
        :param n:
        :return: (x,y) sample in the Euclidean space
        """
        u_rand = random.uniform(0, 1)
        if (u_rand < rce and len(Vg_leaves) is not 0 and not initTree_flag) or self.kdeOpt_flag or self.gmmOpt_flag:# and len(Vg_leaves)>20: #Exploration w.p. 0.5 and explotitation (sampling adaptation) w.p. 0.5
            N_qSamples = 200 #self.params.kSamples*len(self.TreeT.VerticesSet)#np.max([self.params.kSamples*len(self.TreeT.VerticesSet),2*len(self.TreeT.VerticesSet)/self.params.rho])
            t_min = min([vg.curTime for vg in Vg_leaves]) #The fastest trajectory that reach the goal (heuristically)TODO:(Last thing): try min length instead of time
            h = t_min/md

            #Optimality flags, and distribution fitting options
            kdeOpt_flag = self.kdeOpt_flag #The "optimal" kernel density estimate flag
            gmmOpt_flag = self.gmmOpt_flag #The Gaussian mixture model flag
            kde_enabled = self.params.kde_enabled #Kernel fitting istead of EM to fit a gmm
            kde_enabled = self.kde_enabled
            if (kde_enabled and not kdeOpt_flag) or (not kde_enabled and not gmmOpt_flag):
                x,y = self.CE_Sample(Vg_leaves,h,N_qSamples)
            elif kde_enabled and kdeOpt_flag: #The optimal kernel density estimate has been reached, focus the samples just to be for the "optimal" distribution
                # elite_samples_arr = self.kde_currSamples
                xySample = self.KDE_fitSamples.sample()
                x = xySample[0][0]
                y = xySample[0][1]



            if x is None and y is None: #CE_Sample() will return None if not enough samples of trajectoies that reach the goal are avialbe
                x = random.uniform(-1, length)
                y = random.uniform(-1, width)
        else:
            x = random.uniform(-1, length)
            y = random.uniform(-1, width)
        return np.array([x, y])

    def CE_Sample(self,Vg_leaves,h,N_qSamples):
        """
        Exploit the samples of trajectories that reach the goal to adapt the sampling distribution towards the distribution of the rare event.
        The trajectories that reach the goal will be disceretized to extract the elite samples that will be used to adapt (optimize)
        the sampling distribution.

        :param Vg_leaves:
        :param h: The time step to discretize the trajectories
        :param N_qSamples: A threshold indicates the number of samples that are sufficient enough to be exploited (TODO (Doc): How to decide this number)
        :return: None: if the number of points of the discretized trajectories < N_qSamples, (x,y) samples from the estimated distribution
        """
        if len(Vg_leaves)>=(self.adapIter*30): #The acceptable number of trajectories to adapat upon
            frakX = []
            #Find the elite trajectoies then discretize them and use their samples as the elite samples:
            Vg_leaves_costList = [vg.CostToCome for vg in Vg_leaves]
            if (self.adapIter + 3) > 5: 
                d_factor = 15
            elif self.adapIter > 2:
                d_factor = self.adapIter + 3
            else: 
                d_factor = self.adapIter
            q = self.params.rho/d_factor
            cost_rhoth_q = np.quantile(Vg_leaves_costList, q=q)
            elite_Vg_leaves = [vg for vg in Vg_leaves if vg.CostToCome <= cost_rhoth_q]
            if len(elite_Vg_leaves) == 0:
                elite_Vg_leaves = Vg_leaves

            #XXXXXXX
            for vg in elite_Vg_leaves:
                vgCostToCome = vg.CostToCome
                #Concatnating the trajectory:
                traj2vg = np.asarray(vg.StateTraj)
                indexID_anc = vg.ParentIndexID
                while indexID_anc is not None:
                    if indexID_anc is not 0:
                        ParentTraj = self.TreeT.VerticesSet[indexID_anc].StateTraj
                        traj2vg = np.concatenate((ParentTraj,traj2vg),axis=0)
                    indexID_anc = self.TreeT.VerticesSet[indexID_anc].ParentIndexID

                # Backtrack the path from vg to v0; extract the sample at certain increments of the time:
                t0_bktrTraj = timeit.default_timer()
                tStep_init1 = int(h/self.params.step_size)
                tStep_init = 2
                tStep = 10
                tStep_temp = tStep_init1+3
                while tStep < len(traj2vg[:,1]):
                    pi_q_tStep = traj2vg[tStep,:]
                    elite_cddtSample = [pi_q_tStep,vgCostToCome] #This tuple contains the actual sample pi_q_tStep and the CostToCome to the goal of the corresponding trajectory
                    frakX.append(elite_cddtSample)
                    tStep = tStep + tStep_temp
                t1_bktrTraj = timeit.default_timer()
                T_bktrTraj = t1_bktrTraj-t0_bktrTraj
            if self.adapIter == 1:
                frakX.extend(self.initEliteSamples)
            # XXXXXXX
            self.len_frakX = len(frakX)
            if len(frakX) == 0:
                ok = 1
            x,y = self.CE_KDE_Sampling(frakX)
        else:
                x = None
                y = None
        return x,y

    #Density estimate, kernel density or GMM:
    def CE_KDE_Sampling(self,frakX):
        """
        Fit the elite samples to Kernel density estimate (KDE) or a GMM to generate from; and generate an (x,y) sample from the estimated
        distribution. Checks if the CE between the previous density estimate and the current one below some threshold. In the case
        of KDE the expectation similarity measure could be used instead on the CE.

        NOTE to Ahmad:
        You're using the CE with the KDE because you have the logistic probes of the samples and you use them; however, for
        Kernel based distributions the expectation similarity could be used as well. One might reformulate the CE framework
        in terms of nonparametric distributions.

        :param frakX: The elite set of samples with the corresponding trajectory cost.
        :return:
        """
        frakXarr = np.array(frakX)
        N_samples = len(frakX)
        if len(frakXarr.shape) !=2:
            ok =1
        costs_arr = frakXarr[:,1]
        elite_samplesTemp = frakXarr[:,0] #A subset of the samples that are below the elite quantile
        elite_samples = [elite_samplesTemp[i] for i in range(len(elite_samplesTemp))]
        elite_samples_arr = np.asarray(elite_samples)
        elite_costs = costs_arr

        #random point from the estimated distribution:
        if self.kde_enabled:#self.params.kde_enabled:
            #Compute the weight of each elite sample as
            # wSmpl =  1-(cost of the corresponding trajectory of the sample (cost_crsTrajSmpl)/sum(cost_crsTrajSmpl));
            # this weight is the probability of the Sampling Importance Resampling (SIR) of the non-parametric generlized
            # CE method. See Chapter X in the Thesis.

            # sumCost_crsTrajSmpl = sum(elite_costs)
            # w_arr = 1 - elite_costs/sumCost_crsTrajSmpl
            # w_arrNorm = w_arr/sum(w_arr)
            kde = KernelDensity(kernel='gaussian', bandwidth=.85)
            # kde.fit(elite_samples_arr,sample_weight=w_arrNorm)
            kde.fit(elite_samples_arr)
            self.adapIter += 1
            xySample = kde.sample()

        if self.kde_enabled:#self.params.kde_enabled:
            x_gridv = np.linspace(-2, 18, 40)
            y_gridv = np.linspace(-2, 18, 40)
            Xxgrid, Xygrid = np.meshgrid(x_gridv, y_gridv)
            XYgrid_mtx = np.array([Xxgrid.ravel(), Xygrid.ravel()]).T
            #Get the probabilities
            grid_probs = np.exp(kde.score_samples(XYgrid_mtx))

            # Find the KL divergence the current samples and the previous ones:
            if self.adapIter > 2:
                KL_div = self.KLdiv(grid_probs)
                if KL_div < .1:
                    self.kdeOpt_flag = True
                    
                self.KDE_fitSamples = kde #This kde object will be used to sample form whn the optimal sampling distribution has been reached

            self.KDE_pre_gridProbs = grid_probs
            #Save the grid points with the corresponding probs, the cost, and the tree to plot them afterwards:
            saveData(self.goal_costToCome_list, 'adapCBF_RRTstr_Cost', suffix=self.suffix, CBF_RRT_strr_obj=self,
                     adapDist_iter=self.adapIter-1, enFlag=False)

            saveData([self.TreeT,self.vg_minCostToCome_list], 'adapCBF_RRTstr_Tree_CDC', suffix=self.suffix, CBF_RRT_strr_obj=self,
                     adapDist_iter=self.adapIter - 1, enFlag=False)
            saveData([Xxgrid, Xygrid, grid_probs.reshape(Xxgrid.shape),elite_samples_arr], 'adapCBF_RRTstr_KDEgridProbs_CDC',
                     suffix=self.suffix, CBF_RRT_strr_obj=self,
                     adapDist_iter=self.adapIter - 1, enFlag=False)

            #Plot the distribution
            if self.params.plot_pdf_kde:
                self.initialize_graphPlot()
                CS = plt.contour(Xxgrid, Xygrid, grid_probs.reshape(Xxgrid.shape))  # , norm=LogNorm(vmin=4.18, vmax=267.1))
                # plt.colorbar(CS, shrink=0.8, extend='both')
                plt.scatter(elite_samples_arr[:, 0], elite_samples_arr[:, 1])
                plt.show()

        return xySample[0][0],xySample[0][1]

    def treeSample(self):
        """
        Sample a vertex from the expanded tree
        :return: a random vertex from the tree.
        """
        cardV = len(self.TreeT.VerticesSet)
        rand_indexID = 0
        if cardV == 1:
            return self.TreeT.VerticesSet[rand_indexID]
        #Choose a random vertex that is not in Vg_leaves
        vgFlag = True
        while vgFlag:
            rand_indexID = random.randint(0,cardV-1)
            vgFlag = self.TreeT.VerticesSet[rand_indexID].vgFlag
        return self.TreeT.VerticesSet[rand_indexID]

    def KLdiv(self,grid_probs):
        """
        Compute the KL divergence
        :param grid_probs: The probabilities of the point in the grid of the current sampling distribution
        :return: the KL divergence
        """
        if self.kde_enabled:#self.params.kde_enabled:
            pre_grid_probs = self.KDE_pre_gridProbs
        else:
            pre_grid_probs = self.pre_gridProbs
        return -sum([pre_grid_probs[i]*np.log2(grid_probs[i]/pre_grid_probs[i]) for i in range(len(pre_grid_probs))])


    def kdeSimilarity(self):
        """
        Measure the similarity between 2 nonparametric kernel density distributions. Refer for http://users.umiacs.umd.edu/~yangcj/papers/nips2004track.pdf
        for the similarity measure details.
        :return:
        """
        t0_sim = timeit.default_timer()
        X = self.kde_preSamples
        Y = self.kde_currSamples
        K = sklearn.metrics.pairwise.rbf_kernel(X,Y,gamma=1.5)
        sumKs = np.sum(K)
        Jxy = sumKs/(len(X)*len(Y))
        Ldist = -math.log(Jxy)
        t1_sim = timeit.default_timer()
        T_sim = t1_sim - t0_sim
        return Ldist

    ######################################################################################################################:
    ############################## Nearest function #########################################################################:
    def Nearest(self, xy_sample):
        """
        Given the expanded tree, find the nearest vertex to xy_sample. Uses KD-trees as a NN method.
        :param xy_sample: (x,y) in the 2D Euclidean space.
        :return: The nearest vertex from the tree
        """
        if not self.params.kdT_enabled:
            Vertices = self.TreeT.VSet_forNNs
            dlist = [(vertex.State[0] - xy_sample[0]) ** 2 + (vertex.State[1] - xy_sample[1]) ** 2
                     for vertex in Vertices]
            minind = dlist.index(min(dlist))
            nearest_vertex = Vertices[minind]
        else:
            # The position of each node in the mission-space:
            TreeT = self.TreeT #TODO (code optimizaton--mem issue) check if we could not extract it
            if self.params.ordDict_enabled:
                # xyCoorVertices = TreeT.xyCoorVertices.values() #TODO (Q) how about xyCoorVertices = self.TreeT.xyCoorVertices.values() ?
                xyCoorVertices = TreeT.xyVset_forNNs
            else:
                xyCoorVertices = TreeT.xyCoorVertices

            if isinstance(xyCoorVertices, list): #WHY???
                xyCoorVertices = np.array(xyCoorVertices)

            # The tree of the NNs (we cannot build it outside)
            tempTree = KDTree(xyCoorVertices, leaf_size=2) #TODO (meeting) build out of here (when adding a vertex to the tree)

            # The index of the NN vertex (it is different than indexID of the vertex)
            nn_currIndex = int(tempTree.query(xy_sample.reshape([1, 2]), k=1, return_distance=False))
            if not self.params.ordDict_enabled:
                # Return the indexID of the Nearest Vertex to xy_sample:
                indicesID_List = TreeT.VerticesSet.keys()
                nn_indexID = indicesID_List[nn_currIndex]
                nearest_vertex = TreeT.VerticesSet[nn_indexID]
            else:
                nearest_vertex = TreeT.VSet_forNNs[nn_currIndex]
        return nearest_vertex

    ######################################################################################################################:
    ############################## Near function #########################################################################:
    def Near(self,  b_raduis, Vertex = None, xyCoor = None):
        """
        Find all the vertices within a  ball with radius=b_raduis of the Vertex. Uses KDball-trees as well.
        :param b_raduis:
        :param Vertex:
        :param xyCoor:
        :return: A list of the vertexes with the specified ball.
        """
        # The position of each node in the mission-space:
        TreeT = self.TreeT
        # xyCoorVertices = TreeT.xyCoorVertices
        if not self.params.ordDict_enabled:
            xyCoorVertices = TreeT.xyCoorVertices.values()
        else:
            xyCoorVertices = TreeT.xyVset_forNNs

        if isinstance(xyCoorVertices, list):
            xyCoorVertices = np.array(xyCoorVertices)

        # Extract the position of the vertex of interest:
        if Vertex is not None:
            xyVertex = Vertex.State[0:2]
        else:
            xyVertex = xyCoor

        # The tree of the NNs within the ball
        tempTree = BallTree(xyCoorVertices)

        # The index of the NN vertex (it is different than indexID of the vertex)
        nn_currIndex, nn_currDist = tempTree.query_radius(xyVertex.reshape([1, 2]), r=b_raduis, return_distance=True,sort_results=True)  # TODO (debug) prevent the vertex to be neighbor to itself

        nn_currIndex = nn_currIndex[0].astype(int)
        if not self.params.ordDict_enabled:
            # Return the indexID of the Nearest Vertex to xy_sample:
            if len(nn_currIndex) == 1:  # There are no neighbors to the vertex
                pass#raise Exception('The vertex has no NNs within the specified ball radius.')
            else:
                nn_currIndex = nn_currIndex[1:]  # Exclude the node of being its own neighbor

            indicesID_List = TreeT.VerticesSet.keys()
            #nn_indexID = [indicesID_List[i] for i in nn_currIndex.astype(int)]
            nn_verticesSet = [TreeT.VerticesSet[indicesID_List[i]] for i in nn_currIndex]
        else:
            nn_verticesSet = [TreeT.VSet_forNNs[i] for i in nn_currIndex]
        return nn_verticesSet

    # ######################################################################################################################:
    # ############################## SafeSteering Algorithm 2: #############################################################:

    # TODO: creat ba a safeSteeringExact when performing rewiring 
    def SafeSteering(self, v_nearest, desired_theta = None, desired_pos = None , m=10 ,edge_length = 1 ):
        """
        Take the desired theta to steer to (desired_theta); the vertex to steer from (v_nearest);
        the mission space (embed in simObject); and the current tree (embed in self). It will steer with
        m-Steps (m=10) in the  theta direction.

        If RRTstar is eneabled then the function steer in straight lines and check for collosion. 

        :param v_nearest:
        :param desired_theta:
        :param m: The number 
        :param extFlag: if enabled CLF-CBF-QP controller will be used to generate the motion trajectory 
        :return:
        """
        if m<1:
            m=1
        
        # Check if the instruction is to safe steer towards a desired direction [CBF-QP] or to exactly steer to a 
        # desired position [CLF-CBF-QP]: 
        
        # RRT* ext, remove all the obstacles from the sim obj and just generate feasible trajectory: >>> 
        if self.RRTstr_enabled: 
            self.simObject.obsts = [] 
        #<<<<<
        
        if desired_pos is None: #Safe steer [CBF-QP] 
            if self.robotDynType is Dyn.SINGLE_INT:
                u1_ref = np.linspace(edge_length*math.cos(desired_theta),edge_length*math.cos(desired_theta), m)
                u2_ref = np.linspace(edge_length*math.sin(desired_theta),edge_length*math.sin(desired_theta), m)
                u_ref = np.array([u1_ref,u2_ref]).T
            else: 
                vRef = np.linspace(0.5, 0.5, m)  # Assume we're working with the unicycle dynamics
                wRef = np.linspace(desired_theta, desired_theta, m)
                u_ref = np.vstack([vRef, wRef]) #TODO (slow)
            instr = u_ref
        elif desired_theta is None: # Exact Safe steer [CLF-CBF-QP]
            instr = Goal(desired_pos)
        else:
            raise Exception('Control instruction [u_ref or goal] was not provided')

        # if len(wRef) is 0:
        #     desired_theta
        #     raise Exception('check out the desired theta')

        if type(v_nearest)==Vertex:
            xy_v_nearest = v_nearest.State[0:2]  # The starting point of the trajectory
            tInitial = v_nearest.curTime
        else:
            xy_v_nearest = v_nearest
            tInitial = 0
        self.simObject.agents = list()
        self.simObject.u_refs = list()
        self.simObject.cur_timestep = 0
        self.simObject.time_vec = [0]
        # 
        # self.simObject.add_agent(Agent(xy_v_nearest, radius=.4,theta=desired_theta ,instructs=u_ref,dynamics=Dyn.UNICYCLE))
        # self.simObject.add_agent(Agent(xy_v_nearest, radius=.4,theta=desired_theta ,instructs=u_ref,dynamics=self.robotDynType))
        self.simObject.add_agent(Agent(xy_v_nearest, radius=.4,theta=desired_theta ,instructs=instr,dynamics=self.robotDynType))
        qTrajectory, uTrajectory, tTrajectory = self.simObject.initiate()  # TODO: incorporate \theta with the trajectory
        qFinal = qTrajectory[0][-1]
        tFinal = tTrajectory[0][-1] + tInitial
        qTrajectory = qTrajectory[0][1:]
        # temp = self.Col_check(qTrajectory)
        return qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory[0][1:] + tInitial
    
    
    def SafeSteering2Goal(self,xy_agt,xy_goal):
        """
        Generates minimal control trajectory for a unicycle robot defined as single integrator. QP controller is subject
        to CBF-CLF constraints. The CBF constraints for the obstacles, the CLF constraints on the other hand for the goal.
        :return:
        """
        simObject = Simulation()  # TODO (meeting) create a reset option,
        goal = Goal(xy_goal)
        obsList = self.obsWorldList
        simObject.add_agent(Agent(xy_agt, instructs = goal, dynamics = Dyn.UNICYCLE))
        [simObject.add_obstacle(obs) for obs in obsList]
        qTrajectory, uTrajectory, tTrajectory = simObject.initiate()  # TODO: incorporate \theta with the trajectory
        qFinal = qTrajectory[0][:, -1]
        tFinal = tTrajectory[0][-1]
        return qFinal, tFinal, uTrajectory[0], qTrajectory[0], tTrajectory[0]

   

    def ChooseParent(self, Nnear_vSet, v_nearest, v_new):
        """
        Check each vertex in Nnear_vSet, whether if it decrease the costToCome to v_new.
        :param Nnear_vSet: A set of IDs of the vertexes in the neighborhood of v_new
        :param v_nearest: The nearest vertex to v_new
        :param v_new: The vertex that we want to choose better parent for.
        :return: v_min the vertex that reduces the costToCome to v_new among Nnear_vSet
        """
        CBF_RRTstrEnable = self.CBF_RRTstrEnable
        v_min = v_nearest
        c_min = v_nearest.CostToCome + self.Cost_v2v(v_nearest,v_new)
        v_pr = v_new  # This might be changed after the SafeSteering (see below!) Warning use the original v_new value
        xy_v_pr = v_pr.State[0:2]
        if (len(Nnear_vSet) == 1 and Nnear_vSet[0]==v_new) or len(Nnear_vSet)>60:
            pass
        else:
            for v_near in Nnear_vSet:
                if v_near.CostToCome is None:  # Debuggining My friend disregard it!! but even though there must be at least v_nearest in the vicinity
                    v_near = v_nearest
                norm_v_pr2v_near = self.Cost_v2v(v_near,v_new) #This is the Euclidean distance we'll use arcLength afterwards; this is following the assumption
                c_pr_heuristic = v_near.CostToCome + norm_v_pr2v_near #Even though the steed point will have a curved path, its magnitude is the same as the heuristic
                if c_pr_heuristic < c_min:
                    # ---- Compute the desired angle to get to v_pr from the best parent so far (from v_near):
                    mSteering = norm_v_pr2v_near / self.params.step_size  # TODO: check if correct; i.e. prbe the lengthes of the generated trajectories through the safe steering.
                    xy_v_near = v_near.State[0:2]
                    desired_theta = math.atan2(xy_v_pr[1] - xy_v_near[1] , xy_v_pr[0] - xy_v_near[0])

                    if mSteering>1:
                        try: 
                            # qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = self.SafeSteering(v_near,desired_pos=[xy_v_pr[0],xy_v_pr[1]],m=mSteering) #TODO[RSS] Exact safe steering 
                            if self.RRTstr_enabled:
                                qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = self.SafeSteering(v_near,desired_pos=xy_v_pr,m=int(mSteering)+1)
                                steering_okFlag = True
                                coll_flag = self.iscoll(qTraj=qTrajectory)
                                if np.linalg.norm(xy_v_pr - qFinal[0:2]) > .1 or coll_flag:
                                    steering_okFlag = False
                            else:
                                # qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = self.SafeSteering(v_near,desired_theta=desired_theta,m=int(mSteering)+1)
                                qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = self.SafeSteering(v_near,desired_pos=xy_v_pr,m=int(mSteering)+1)
                                steering_okFlag = True
                                # steering_okFlag = isAcceptableSample(traj=qTrajectory, desiredSteps_num = int(mSteering)+1)
                                if np.linalg.norm(xy_v_pr - qFinal[0:2]) > .1:
                                    steering_okFlag = False
                        except:
                            steering_okFlag = False
                    else:
                        steering_okFlag = False

                    if steering_okFlag:
                        v_min = v_near
                        v_pr.State = qFinal
                        v_pr.StateTraj = qTrajectory
                        v_pr.timeTraj = tTrajectory
                        norm_v_pr2v_near = self.Cost_v2v(v_near,v_new)  # This is the Euclidean distance we'll use arcLength afterwards; this is following the assumption
                        c_pr_heuristic = v_near.CostToCome + norm_v_pr2v_near
                        v_pr.CostToCome =  c_pr_heuristic
                        if len(v_pr.StateTraj) is 1:
                           v_pr
                        v_pr.ParentIndexID = v_near.indexID
                        self.updateVertex(v_pr)
                        c_min = c_pr_heuristic
                    else:
                        continue #Try another v_near
        if v_pr.ParentIndexID is None:
            v_pr.ParentIndexID=v_nearest.indexID
            v_pr.CostToCome = c_min
        return v_min, v_pr

    def Cost_v2v(self, v_start, v_end):
        """
        Compute the unit norm distance between v_start and v_end. It is faster to compute the norm this way (why?).
        :param v_start:
        :param v_end:
        :return: The unit norm distance
        """
        xy_v_start = v_start.State[0:2]
        xy_v_end = v_end.State[0:2]
        EucNorm = np.sqrt((xy_v_end[0]-xy_v_start[0])**2+(xy_v_end[1]-xy_v_start[1])**2)
        return EucNorm

    def Rewire(self, Nnear_vSet, v_nearest, v_new):
        """
        For each vertex with ID in Nnear_vSet check if assigning v_new as a parent for that vertex reduce the costToCome to it.

        :param Nnear_vSet: the same as in ChooseParent
        :param v_nearest:
        :param v_new:
        :return:
        """
        #TODO[RSS]: Perform SafesteeringExact when rewiring (before macking this change, check if the position of the rewired vertex has deviated)
        v_pr = v_new
        CBF_RRTstrEnable  = self.CBF_RRTstrEnable
        if len(Nnear_vSet)>60:
            return
        for v_near in Nnear_vSet:
            #The heuristic CostToCome to v_near through v_new (i.e. after the the rewire attempt)
            norm_v_pr2v_near = self.Cost_v2v(v_pr, v_near)
            c_rewired_heuristic = v_new.CostToCome + self.Cost_v2v(v_new,v_near)
            c_preRewired = v_near.CostToCome

            #Check if making v_new a parent would decrease the CostToCome: TODO: (check the time)
            if  c_rewired_heuristic<c_preRewired: #Heuristicly the path after rewiring yielded a lower cost than the original COstToCome
                # The indexID of the current parent (will be updated after this condition) of v_near; in order to update
                # its ChildrenIndexIDs set:
                prev_ParentIndexID = v_near.ParentIndexID

                #Preparing to steer to v_near after the rewring attempt; this step is to check the actual cost (arcLength) and where the SafeSteer has ended up
                xy_v_near = v_near.State[0:2]
                xy_v_new  = v_new.State[0:2]
                mSteering = norm_v_pr2v_near / self.params.step_size
                desired_theta = math.atan2(xy_v_near[1]-xy_v_new[1] , xy_v_near[0]- xy_v_new[0] )

                # SafeSteer from v_new to v_near (rewire v_near to a better parent vertex)
                if mSteering > 1:
                    try:
                        if self.RRTstr_enabled:
                            qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = self.SafeSteering(v_new,
                                                                                                    desired_pos=[xy_v_near[0],xy_v_near[1]],
                                                                                                    m=mSteering) # TODO[RSS] Exact safe steering
                            steering_okFlag = True ###Major bug! steering could be ok though the final position is not.
                            coll_flag = self.iscoll(qTraj=qTrajectory)
                            if np.linalg.norm(xy_v_near - qFinal[0:2]) > .1 or coll_flag:
                                steering_okFlag = False
                        else:
                            qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = self.SafeSteering(v_new,
                                                                                                    desired_pos=xy_v_near,
                                                                                                    m=mSteering) # TODO[RSS] Exact safe steering
                            # qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = self.SafeSteering(v_new,
                            #                                                                         desired_theta=desired_theta,
                            #                                                                         m=int(mSteering)+1)
                            steering_okFlag = isAcceptableSample(traj=qTrajectory, desiredSteps_num=int(mSteering)+1) ###Major bug! steering could be ok though the final position is not.
                            steering_okFlag = True
                            if np.linalg.norm(xy_v_near - qFinal[0:2]) > .1:
                                steering_okFlag = False
                    except:
                        steering_okFlag = False

                else:
                    steering_okFlag = False
                if True:
                        if steering_okFlag:
                            try:
                                self.TreeT.VerticesSet[prev_ParentIndexID].ChildrenIndexIDs.remove(v_near.indexID)
                                #v_near.updateParent(parentVertex=v_new)
                                v_near.ParentIndexID = v_new.indexID
                                prewiredState = v_near.State
                                v_near.State = qFinal
                                v_near.curTime = tFinal #TODO (cbfRRT*)
                                v_near.CtrlTraj = uTrajectory #TODO (cbfRRT*)
                                prewiredTrajectory = v_near.StateTraj
                                v_near.StateTraj = qTrajectory #TODO (cbfRRT*)
                                v_near.timeTraj = tTrajectory #TODO (cbfRRT*)
                                if len(v_near.StateTraj[0,:]) is 1:
                                    v_near
                                self.addChild(vParent=v_new,vChild=v_near)
                            except:
                                continue
                        else:
                            continue #The node must not be rewired!!!
                #Compute c_rewired_heuristic after the safesteering action:
                c_rewired_heuristic = v_new.CostToCome + self.Cost_v2v(v_new,v_near)
                v_near.CostToCome = c_rewired_heuristic
                if v_new.ParentIndexID == v_near.indexID:
                        raise Exception('That is not')

                #Update the updated vertex in the tree of the CBF_RRT_strr class
                self.updateVertex(v_near)

                #Update the cost of the children of v_near:
                self.UpdateCostToCome(parentVertex= v_near,c=v_near.CostToCome)

    def UpdateCostToCome(self,parentVertex,c):
        """
        Update the costToCome to a child vertex of a vertex that its costToCome has been changed.
        :param parentVertex:
        :param c: The costToCome to parentVertex
        :return: -
        """
        parentVertexCostToCome = parentVertex.CostToCome
        if parentVertexCostToCome>=c and parentVertex.ChildrenIndexIDs is not None:
            for childIndexId in parentVertex.ChildrenIndexIDs:
                childVertex = self.TreeT.VerticesSet[childIndexId]
                if self.params.CBF_RRTstrEnable:
                    timeDiff = childVertex.timeTraj[0] - parentVertex.curTime
                    childVertex.timeTraj = childVertex.timeTraj - timeDiff
                self.TreeT.VerticesSet[childIndexId].CostToCome = c+self.Cost_v2v(v_start=parentVertex,v_end=childVertex)
                if self.TreeT.VerticesSet[childIndexId].ChildrenIndexIDs is not None:
                    self.UpdateCostToCome(parentVertex=self.TreeT.VerticesSet[childIndexId],c=c+self.Cost_v2v(v_start=parentVertex,v_end=childVertex))

    def UpdateCostToGo(self,vertex,cost):
        """
        Update the costToGo from vertex to the goal region.
        :param vertex: The vertex in which we will update its cost-to-go as well as its ancestors
        :param cost: costToGo to the goal vertex from vertex
        :return: -
        """
        vertex_indexID = vertex.indexID
        currCostToGo = vertex.CostToGo
        while currCostToGo > cost:

            #Set the cost-to-go for the current vertex (the cost-to-go for the 1st vertex is fed directly)
            self.TreeT.VerticesSet[vertex_indexID].CostToGo = cost
            ParentIndexID = self.TreeT.VerticesSet[vertex_indexID].ParentIndexID
            if ParentIndexID is None:
                break
            Parent2ChildCost = self.TreeT.VerticesSet[vertex_indexID].CostToCome - self.TreeT.VerticesSet[ParentIndexID].CostToCome
            cost = cost + Parent2ChildCost

            #Update for the next iteration; assign the parent vertex to be updated for in the next iteration:
            currCostToGo = self.TreeT.VerticesSet[ParentIndexID].CostToGo
            vertex_indexID = ParentIndexID

    def iscoll(self,xy_sample = None, qTraj = None):
        """
        Returns True if xy_sample or any point of qTraj is in collosion. 
        
        Input: 
        - xy_sample: 
        - qTraj: 
        """
        col_flag = False
        obsWorldList = self.obsWorldList
        if xy_sample is not None: 
            for obst in obsWorldList: 
                try:
                    M_ellip = obst.M_ellip
                except: 
                    raise('Define your obstacle as an ellipsoid (even if it is a sphere)')
                h = ((xy_sample - obst.state[0:2]).transpose().dot(M_ellip).dot((xy_sample - obst.state[0:2])) - 1) * obst.sign
                if h<0:
                   col_flag = True
                   break 
        elif qTraj is not None:
            # TODO: make a list of xyTraj of the xy positions of the trajectory: 
            for xy in qTraj:
                for obst in obsWorldList: 
                    try:
                        M_ellip = obst.M_ellip
                    except: 
                        raise('Define your obstacle as an ellipsoid (even if it is a sphere)')
                    h = ((xy - obst.state[0:2]).transpose().dot(M_ellip).dot((xy - obst.state[0:2])) - 1) * obst.sign
                    if h<0:
                        col_flag = True
                        break
        else:
            raise('Neither xy_sample nor qTraj are passed to the function')
        return col_flag
        

    def MiniTime_Traj(self):
        pass


    def updateVertex(self,VertexToUpdate):
        """Update the vertex in the tree"""
        TreeT = self.TreeT
        TreeT.updateVertex(VertexToUpdate)
        self.TreeT = TreeT
    def addVertex(self, VertexToAdd):
        TreeT = self.TreeT
        TreeT.addVertex(VertexToAdd)
        self.TreeT = TreeT
    def addChild(self,vParent,vChild):
        """Add the indexID of vChild to the ChildrenIndexIDs of vParent"""
        ChildindexID = vChild.indexID
        vParent.addChild(ChildindexID)
        self.updateVertex(vParent)

    def addGoalVertex(self,vGoal):
        self.Vg_leaves.append(vGoal)


    #===================================================================================
    #>>>>>>>>>>>>>>>>>>>>>>>> The main method to perform the CBF-RRT* <<<<<<<<<<<<<<<<<<
    #===================================================================================
    def Plan_CBF_RRT_strr(self,initTree_flag = False):
        """
        Generate an RRT* that reaches the goal region.
        If self.params.CBF_RRTstrEnable the tree will be with the safeSteering, otherwise the tree will just as the
        regular RRT*.

        :param initTree_flag: If enabled the generation of the tree will ignore the obstacles in the workspace.
        :return: v_final ,self.TreeT
        """

        ball_steeringSteps = int(self.params.edge_length/self.params.step_size)
        b_radi = self.params.edge_length
        #NSampling = self.NSampling  # Maximum # of samples
        reachedFlag = False
        CBF_RRTstrEnable = self.CBF_RRTstrEnable
        i = 1
        actual_i = 1
        goalTrajCost = []

        # if not initTree_flag:
        #     self.initialize_graphPlot()
        xy_goal = self.xy_goal

        #Extracting the CE parameters:
        length = self.params.length
        width = self.params.width
        rho = self.params.rho
        rce = self.params.rce
        md  = self.params.md
        length = self.params.length
        width = self.params.width
        # ------------------------------------------------------------------------------------------------------------------
        #For time analysis:
        t00 = timeit.default_timer()

        goal_costToCome_list = []
        iterTime_list = []
        vg_minCostToCome_list = []
        goal_tCost_list = []

        while i <= self.NSampling or not (reachedFlag): #TODO change
            
            if self.CBF_CLF_enabled: 
                qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = self.SafeSteering(self.TreeT.VerticesSet[0],
                                                                                         desired_pos=[xy_goal[0],xy_goal[1]])
                saveData([qTrajectory], self.prefix+'CLF_CBF_traj_rss', suffix=self.suffix,CBF_RRT_strr_obj=self,
                         adapDist_iter=None, enFlag=False)
                break
            # Adapt the sampling distribution based on the expanded trajectories to the goal are
            # ===== Generate Sample in the workspace space and find the nearest vertex to it:
            
            xy_sample = self.UpSampling_andSample(self.Vg_leaves,md=md,rho=rho,\
                rce=rce,length=length,width=width,initTree_flag=initTree_flag)  # Until this moment this method returns a sample in the x-y space
            #>>>> RRTstar_ext (Return a sample in the free workspace):
            
            if self.RRTstr_enabled: 
                coll_flag = self.iscoll(xy_sample)
                # If the sample is collision free, increment the sample iteration (smpl_tr) and the acceptable samples (smpl_axpd)
                if coll_flag: # The sample is in collision  
                    self.smpl_tr = self.smpl_tr + 1    
                    continue
                else:
                    self.smpl_tr = self.smpl_tr + 1
                    self.smpl_axpd = self.smpl_axpd + 1     
            else: 
                self.smpl_tr = self.smpl_tr + 1
                self.smpl_axpd = self.smpl_axpd + 1

            v_nearest = self.Nearest(xy_sample)  # Return the indexID of the NN vertex
            xy_v_nearest = v_nearest.State[0:2]
            desired_theta = math.atan2( xy_sample[1]-xy_v_nearest[1],xy_sample[0]-xy_v_nearest[0])
            ball_steeringSteps = b_radi / self.params.step_size

            # ===== Steering from xy_v_nearest towards xy_sample with M-steps (to be specified):

            # TODO[RSS] No need for this safe-steering step, you just need to safe steer in chose parent. Given that choose parent procedure will return the nn vertex anyway. 
            if self.RRTstr_enabled:
                xy_d = [xy_v_nearest[0]+(b_radi*math.cos(desired_theta)),xy_v_nearest[1]+(b_radi*math.sin(desired_theta))] 
                qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = \
                    self.SafeSteering(v_nearest, desired_pos= xy_d,m=ball_steeringSteps)
                coll_falg = self.iscoll(qTraj=qTrajectory)
                if coll_flag: # The edge is in collision  
                    i = i+1
                    self.i = i
                    continue
                v_new = Vertex(State=qFinal, StateTraj=np.asarray(qTrajectory), CtrlTraj=uTrajectory,\
                        timeTraj=tTrajectory,curTime=tFinal, indexID=i)
            else:
                qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = \
                    self.SafeSteering(v_nearest, desired_theta = desired_theta,m=ball_steeringSteps) #TODO (cbfRRTst)
                # sample_okFlag = isAcceptableSample(traj = qTrajectory, desiredSteps_num = ball_steeringSteps)
                # if sample_okFlag: # Why we need it? 
                v_new = Vertex(State=qFinal, StateTraj=np.asarray(qTrajectory), CtrlTraj=uTrajectory,\
                        timeTraj=tTrajectory,curTime=tFinal, indexID=i) #TODO (cbfRRT*)
                # else:
                #     continue
        
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


            
            #Since we will have more artificial samples to adapt the sampling distribution:
            # if len(v_new.StateTraj[:,0]) is 1:
            #     actual_i = actual_i + 1
            actual_i = actual_i + 1
            self.addVertex(v_new)
            # ===== Check the near vertices in the tree within a ball of the vertex:
            gamma = self.params.gamma
            # gamma = 20
            #Experemntally for edge length =1
            cardV = len(self.TreeT.VerticesSet)                  # Complexity: O(1)
            vb_radi = gamma * math.sqrt(math.log(cardV) / cardV) #gamma * math.log(cardV)
            steer_radi = (self.params.edge_length)+.15
            b_radi = min(vb_radi, steer_radi) #TODO why it is slow?
            # b_radi = .95
            if b_radi<.46:
                b_radi = .46


            #RRT
            # Nnear_vSet = self.Near(Vertex=v_new,b_raduis=b_radi)  # TODO: (debug) prevent the vertex to be neighbor to itself
            # Nnear_vSet = Nnear_vSet[1:]

            # ===== Choose the best parent for v_new from the near vertices:
            #RRT
            if not self.CBF_RRT_enable: 
                # b_radi = min(vb_radi, steer_radi)
                Nnear_vSet = self.Near(Vertex=v_new,b_raduis=b_radi+1)  # TODO: (debug) prevent the vertex to be neighbor to itself
                Nnear_vSet = Nnear_vSet[1:]
                # v_nearest = Nnear_vSet[0] 
                v_min, v_new = self.ChooseParent(Nnear_vSet=Nnear_vSet, v_nearest=v_nearest, v_new=v_new)
            else: # No rewiring procedure
                v_min = v_nearest
                v_new.ParentIndexID = v_min.indexID
                v_new.CostToCome = v_min.CostToCome + self.Cost_v2v(v_nearest,v_new)
            self.addChild(v_min,v_new) # (v_parent,v_child)


            # ===== Rewire the near vertices (check if choosing v_new as a parent would lower the cost)
            if not self.CBF_RRT_enable: #No rewiring procedure is required: 
                # b_radi = min(vb_radi, steer_radi)
                # Nnear_vSet = self.Near(Vertex=v_new,b_raduis=b_radi)  # TODO: (debug) prevent the vertex to be neighbor to itself
                # Nnear_vSet = Nnear_vSet[1:]
                self.Rewire(Nnear_vSet=Nnear_vSet, v_nearest=v_min, v_new=v_new)


            # ===== Inspecting v_goal =============
            if not self.params.dataGen_enabled:
                if True:#(self.initTree_done) or (self.params.kde_enabled and not self.params.CE_enabled):
                    vg_nearSet = self.Near(xyCoor=xy_goal, b_raduis=self.eps_g)
                    if len(vg_nearSet) > 0:
                        if not self.goalReached:
                            self.goalReached = True
                            self.iGoalReached = i
                        costMin = 10000
                        for vertex in vg_nearSet:
                            if vertex.CostToCome < costMin and vertex.CostToCome is not None:
                                vg_minCostToCome = vertex
                                costMin = vg_minCostToCome.CostToCome

                        # if self.vg_minCostToCome_list is not None:
                        #     vg_minCostToCome = self.vg_minCostToCome_list[0]
                        #     costMin = vg_minCostToCome.CostToCome
                        # if costMin > self.bestGCost:
                        #     a = 1
                        # self.bestGCost = costMin
                        # vg_minCostToCome = vg_nearSet[0]

                        # $$$$$$$$$$$
                        # Traj,timeTraj =  self.get_xythetaTraj(vg_minCostToCome)
                        # plt.plot(timeTraj,Traj[0,:])
                        # plt.show()
                        # plt.plot(timeTraj, Traj[1, :])
                        # plt.show()
                        # plt.plot(timeTraj, Traj[2, :])
                        # plt.show()

                        # $$$$$$$$$$$

                        gCostToCome = vg_minCostToCome.CostToCome
                        if vg_minCostToCome.CostToCome is None:
                            if self.goal_costToCome_list is not None:
                                iCost = len(self.goal_costToCome_list) - 1
                                while gCostToCome is None:
                                    gCostToCome = self.goal_costToCome_list[iCost]
                                    iCost = iCost - 1
                        else:
                            goal_costToCome_list.append(gCostToCome)
                            vg_minCostToCome_list.append(vg_minCostToCome)
                            self.goal_costToCome_list = goal_costToCome_list
                            self.vg_minCostToCome_list = vg_minCostToCome_list  # The list of vgoal
                            goal_tCost_list.append(vg_minCostToCome.curTime)

                        #Record the trajectory:
                        # Traj, _ = self.get_xythetaTraj(vg_minCostToCome)
                        Traj = []
                        # Plotting the evolution of the costs:
                        try:
                            if False: #actual_i % 100 == 0:#actual_i % 50 == 0 and vg_minCostToCome.CostToCome is not None:
                                Traj, timeTraj = self.get_xythetaTraj(vg_minCostToCome)
                                timeTraj = np.linspace(0, len(Traj[0, :]) * self.params.step_size, len(Traj[0, :]))
                                # plt.plot(timeTraj, Traj[0, :])
                                # plt.show()
                                # plt.plot(timeTraj, Traj[1, :])
                                # plt.show()
                                # plt.plot(timeTraj, Traj[2, :])
                                # plt.show()

                                # self.initialize_graphPlot()
                                # plt.plot(Traj[0, :], Traj[1, :])
                                # plt.show()

                                # plt.plot(np.linspace(1, len(goal_tCost_list), len(goal_tCost_list)), goal_tCost_list,
                                #         label="Time Cost", hold='on')
                                # plt.xlabel("number of vertexes ", fontsize=15)
                                # plt.ylabel("time cost", fontsize=15)
                                # plt.xticks(size=10)
                                # plt.yticks(size=10)
                                # plt.grid(True)
                                # plt.show()

                                plt.plot(np.linspace(self.iGoalReached, self.iGoalReached + len(goal_costToCome_list),
                                                    len(goal_costToCome_list)), goal_costToCome_list, label="Cost-to-come",
                                        hold='on')
                                plt.plot(np.linspace(self.iGoalReached, self.iGoalReached + len(goal_costToCome_list),
                                                    len(goal_costToCome_list)),
                                        np.repeat(22.896, len(goal_costToCome_list)))
                                plt.legend(['The proposed algorithm', 'CBF-CLF QP'])
                                plt.xlabel("number of vertexes ", fontsize=15)
                                plt.ylabel("cost-to-come", fontsize=15)
                                plt.xticks(size=10)
                                plt.yticks(size=10)
                                plt.grid(True)
                                plt.show()
                        except:
                            pass 

                    else:
                        vg_minCostToCome = None
                else:
                    vg_minCostToCome = None

                # Plotting the expansion tree
                if False:#actual_i % 50 == 0:  # actual_i == 10 or actual_i == 20 or actual_i==100 or actual_i==150 or actual_i==200 or actual_i==1000:
                    t1 = timeit.default_timer()
                    xy_plot = self.xy_goal
                    self.initialize_graphPlot()
                    goalVertex = self.plot_tree(vg_minCostToCome, plot_pathFalg=True)
                    self.TreePlot.show()
            else:  # Save the tree as a Mat, the cost as a mat file
                pass

            # #----------------------------------------------------------------------------------------

            #Return the first vertex that reaches the goal for the initial tree:
            if initTree_flag:
                xyv_new = v_new.State[0:2]
                if np.linalg.norm(xy_goal-xyv_new) < 1.8:
                    v_final = v_new
                    break
            else:
                v_final = None



            #Probing the tree costs at 100, 500, 2500, and 10000 iterations:
            flag100   = (self.i==99 or self.i==100 or self.i==101)
            flag300   = (self.i == 299 or self.i == 300 or self.i == 301)
            flag500   = (self.i == 499 or self.i == 500 or self.i == 501)
            flag1000  = (self.i == 999 or self.i == 1000 or self.i == 1001)
            flag2000  = (self.i == 1999 or self.i == 2000 or self.i == 2001)
            flag3000  = (self.i == 2999 or self.i == 3000 or self.i == 3001)
            flag5000  = (self.i == 4999 or self.i == 5000 or self.i == 5001)
            flag10000 = (self.i == 9999 or self.i == 10000 or self.i == 10001)
            if flag3000:# flag500 or flag1000 or flag2000 or flag5000:#flag10000 or flag1000 or flag2500 or flag5000 or flag100 or flag500:
                #Save the costs:

                #The first element of the saved list contains the iteration that the goal has been reached at.
                saveData([self.iGoalReached,self.goal_costToCome_list], self.prefix+'CBF_RRTstr_Cost_CDC', suffix=self.suffix,CBF_RRT_strr_obj=self,
                         adapDist_iter=None, enFlag=True)

                saveData(self.iterTime_list, self.prefix + 'CBF_RRTstr_iterTime_CDC',
                         suffix=self.suffix, CBF_RRT_strr_obj=self,
                         adapDist_iter=None, enFlag=False)

                #Save the tree:
                saveData([self.TreeT,self.vg_minCostToCome_list], self.prefix+'CBF_RRTstr_Tree_CDC', suffix=self.suffix, CBF_RRT_strr_obj=self,
                         adapDist_iter=None, enFlag=False)
                saveData([Traj], self.prefix + 'CBF_RRTstr_PathToG', suffix=self.suffix,
                         CBF_RRT_strr_obj=self,
                         adapDist_iter=None, enFlag=False)

            if i>3000:
                print("done")
                break
            print("Iter:", i)
            #----------------------------------------------------------------------------------------------------------
            #----------------------------------------------------------------------------------------------------------
            #--------------------- CE-extension:
            #RRT
            ParentIndexIDv_new = v_new.ParentIndexID
            vParent_new = self.TreeT.VerticesSet[ParentIndexIDv_new]
            kdeOpt_flag = self.kdeOpt_flag
            gmmOpt_flag = self.gmmOpt_flag
            kde_enabled = self.params.kde_enabled
            kde_enabled = self.kde_enabled
            CE_enabled = self.params.CE_enabled
            if ((kde_enabled and not kdeOpt_flag) or (CE_enabled and not gmmOpt_flag)) and (self.initTree_done or not self.params.initTree_flag) and not vParent_new.vgFlag:# and cardV >30:
                #Attempt steering to the goal:
                xy_v_new = v_new.State[0:2]
                mSteering2goal = np.linalg.norm(xy_goal-xy_v_new) / (self.params.step_size*1.5)

                theta_goal = math.atan2(xy_goal[1] - xy_v_new[1], xy_goal[0] - xy_v_new[0])
                qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = self.SafeSteering(v_new,
                                                                                          desired_theta=theta_goal,
                                                                                          m=int(mSteering2goal))
                # qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = self.SafeSteering(v_new,
                #                                                                           desired_pos=[xy_goal[0],xy_goal[1]],
                #                                                                           m=int(mSteering2goal)+1)
                if mSteering2goal<1:
                    a = 1
                if len(qTrajectory) != len(tTrajectory):
                    a = 1

                #Check how if we ended up away from the goal:
                if np.linalg.norm(xy_goal-qFinal[0:2]) <= self.eps_g+.8:
                    i = i + 1 #The goal vertex will be added anyways to the tree; given that during the initialization we have grown a tree reaches the goal
                    v_goal = Vertex(State=qFinal, StateTraj=qTrajectory, CtrlTraj=uTrajectory, timeTraj=tTrajectory,
                                   curTime=tFinal, indexID=i,ParentIndexID=v_new.indexID,CostToGo = 0.,vgFlag=True)  # TODO (cbfRRT*)
                    v_newCostToGo = self.Cost_v2v(v_start=v_new, v_end=v_goal)
                    v_goal.CostToCome = v_new.CostToCome + v_newCostToGo
                    if abs(v_goal.CostToCome-(v_new.CostToCome + v_newCostToGo)) > 0.1:
                        v_goal.CostToCome
                    self.addVertex(VertexToAdd=v_goal)
                    self.addChild(vParent=v_new,vChild=v_goal)
                    self.addGoalVertex(vGoal = v_goal)
                    v_newCostToGo = self.Cost_v2v(v_start=v_new,v_end=v_goal)
                    self.UpdateCostToGo(vertex=v_new,cost=v_newCostToGo)
                    i = i + 1

                else:
                    i = i + 1
            else:
                i = i+1
            self.i = i
            if i>10000:
                a=1

            
            

        return v_final ,self.TreeT

    def initialize_graphPlot(self):
        """Return initial figure object with the mission space (dimension and obstacles) """

        # Figure parameters:
        axislim = [-5, self.params.length, -5, (self.params.width)]
        colors = np.array(
            [[0., 1., 0.], [1., 0., 1.], [0., 0., 1.], [0., 1., 0.], [1., 1., 0.], [0., 0., 0.], [1., 0., 0.]])
        fig, ax = plt.subplots()

        # Plot the obstacles:
        obsList = self.obsWorldList
        if len(obsList) > 0 and not None:
            for obs in obsList:
                ax = plt.gca()
                obs.plot(ax)
        # Plot the start and goal locations:
        xy_start = self.q_init[0:2]
        xy_goal = self.xy_goal[0:2]
        eps_g = self.eps_g
        eps_g = 0.15
        
        goalRegion = plt.Circle((xy_goal[0], xy_goal[1]), eps_g, linestyle='--', color='g', fill=False)
        ax.add_patch(goalRegion)
        plt.plot(xy_start[0], xy_start[1], "xb", markersize=15, label="Initial State", hold='on')
        plt.plot(xy_goal[0], xy_goal[1], "^r", markersize=15, label="Goal State", hold='on')

        plt.xlabel("$x$", fontsize=20)
        plt.ylabel("$y$", fontsize=20)
        plt.xticks(size=15)
        plt.yticks(size=15)
        # plt.axis(axislim)
        plt.grid(True)
        plt.legend(loc='best', borderaxespad=0., prop={'size': 15})
        plt.rcParams.update({'font.size': 15})

        self.TreePlot = plt


    def plot_vertex(self, vertex, EnPlotting=False):
        """Plot the vertex location in the 2D space. Will extract the xy_pose and plot accordingly.
           Inputs:
            - vertex: Vertex object
            - EnablePlotting: Plot is true, pass the function otherwise
           Return/Update:
             The TreePlot attribute of the CBF_RRTstrr object will be updated.
             Showing the plot will be done explicitly outside the function.
         """
        if EnPlotting:
            plt = self.TreePlot
            xy_vertex = vertex.State[0:2]
            plt.plot(xy_vertex[0], xy_vertex[1], "o", markersize=.5, color=[0., 1., 0.], mfc=[0., 1., 0.],
                     mec=[0.1, 0.1, 0.1], hold='on')
            self.TreePlot = plt
        else:
            pass


    def plot_pathToVertex(self, vertex,marker = None, EnPlotting=False, colorPath=[1., 0., 1.]):
        """Plot the vertex location in the 2D space. Will extract the xy_pose and plot accordingly.
            Inputs:
            - vertex: Vertex object
            - EnablePlotting: Plot is true, pass the function otherwise
            Return/Update:
            The TreePlot attribute of the CBF_RRTstrr object will be updated.
            Showing the plot will be done explicitly outside the function.
         """
        if EnPlotting:
            plt = self.TreePlot
            if type(vertex) == Vertex:
                vertexTrajectory = vertex.StateTraj
            else:
                vertexTrajectory = vertex
            vertexTrajectory = np.asarray(vertexTrajectory)
            if marker is None: 
                plt.plot(vertexTrajectory[:, 0], vertexTrajectory[:, 1], markersize=.1, color=colorPath, mfc=[1., 0., 1.],
                     mec=[0.1, 0.1, 0.1], hold='on')
            else: 
                plt.plot(vertexTrajectory[:, 0], vertexTrajectory[:, 1],marker, markersize=5, color=colorPath, mfc=[1., 0., 1.],
                     mec=[0.1, 0.1, 0.1], hold='on')
            self.TreePlot = plt
        else:
            pass


    def plot_tree(self,v_nearest2point = None,plot_pathFalg = False):
        VerticesSet = self.TreeT.VerticesSet
        VerticesSetValues = VerticesSet.values()

        # Plot all the paths
        for vertex in VerticesSetValues:
            if False:# vertex.vgFlag and not self.params.plt_ex2g_Es:
                continue
            else:
                self.plot_vertex(vertex, EnPlotting=True)
                if vertex.ParentIndexID is not None:
                    self.plot_pathToVertex(vertex, EnPlotting=True)
                    # w = vertex.StateTraj[0,0]-self.TreeT.VerticesSet[vertex.ParentIndexID].State[0]
                    # if w>.2:
                    #     a = 2

        # Plot the Path to the closest vertex to xy_point:
        if plot_pathFalg and v_nearest2point is not None:
            # Nnear_vSet = self.Near(Vertex=v_nearest2point, b_raduis=2)
            # if Nnear_vSet[0].CostToCome is None:
            #     return v_nearest2point
            # v_min, v_new = self.ChooseParent(Nnear_vSet=Nnear_vSet, v_nearest=Nnear_vSet[0], v_new=v_nearest2point)
            currVertex = v_nearest2point
            while currVertex.indexID is not 0:
                self.plot_pathToVertex(currVertex, EnPlotting=True, colorPath=[0., 1., 0.])
                currVertex = VerticesSet[currVertex.ParentIndexID]
            return v_nearest2point

    
   
    def plot_Path2Goal(self,trees = None,v_nst2ptrrt = None,v_nst2ptrrt_star = None,v_nst2pt_adrrt_star = None,clfcbf_traj = None,plot_pathFalg = False):
        
    
        #Plot CBF-RRT
        currVertex = v_nst2ptrrt
        VerticesSet = trees[0]
        while currVertex.indexID is not 0:
            self.plot_pathToVertex(currVertex,marker = 'r--', EnPlotting=True, colorPath=[0., 1., 0.])
            currVertex = VerticesSet[currVertex.ParentIndexID]
        
        #Plot CBF-RRTstar
        currVertex = v_nst2ptrrt_star
        VerticesSet = trees[1]
        while currVertex.indexID is not 0:
            self.plot_pathToVertex(currVertex,marker = 'bs', EnPlotting=True, colorPath=[1., 0., 0.])
            currVertex = VerticesSet[currVertex.ParentIndexID]
        
        #Plot adaptive rrt star
        currVertex = v_nst2pt_adrrt_star
        VerticesSet = trees[2]
        while currVertex.indexID is not 0:
            self.plot_pathToVertex(currVertex, EnPlotting=True, colorPath=[0., 0., 1.])
            currVertex = VerticesSet[currVertex.ParentIndexID]
            
        #Plot CLF-CBFQP path 
        clfcbf_traj = np.asarray(clfcbf_traj)
        plt.plot(clfcbf_traj[0,:, 0], clfcbf_traj[0,:, 1],'g^', markersize=5, color=[1.,1.,0.], mfc=[1., 0., 1.],\
                     mec=[0.1, 0.1, 0.1])
        a = 2

    def get_xythetaTraj(self,v):
        """

        :param v: The vertex that we want to trac its x-trajectory
        :return:
        """
        VerticesSet = self.TreeT.VerticesSet
        currVertex = v
        Traj_v02v = np.array([[],[]])
        timeTraj_v02v = np.array([])
        while currVertex.indexID is not 0:

            #The states' trajectory:
            currVertexTraj = currVertex.StateTraj
            currVertexTraj = np.asarray(currVertexTraj) #We flip the trajectory because we're tracing form the end point to the start point
            currVertex_timeTraj = currVertex.timeTraj
            # currVertex_timeTraj = np.flip(currVertex_timeTraj,0)

            #Append the trajectories:
            if len(Traj_v02v[0]) is 0:
                Traj_v02v = currVertexTraj
            else:
                Traj_v02v = np.concatenate((Traj_v02v,currVertexTraj),axis=0)
            timeTraj_v02v = np.concatenate((timeTraj_v02v, currVertex_timeTraj), axis=0)

            currVertex = VerticesSet[currVertex.ParentIndexID]

        return np.flip(Traj_v02v,1),np.flip(timeTraj_v02v,0)
#TODO(No need) typically the arc length = euclidean distance; the SafeSteer will will move the requested number of steps anyway
def saveData(dataAsList,prefix,suffix,CBF_RRT_strr_obj,adapDist_iter = None,enFlag = False):
    # Creat a Figures folder and get the directory name:
    if not os.path.exists('OutputData'):
        os.makedirs('OutputData')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'OutputData/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #Create the name of the datafile:
    if adapDist_iter is None:
        fileName = prefix + "_iter" + str(CBF_RRT_strr_obj.i)+ "_" + suffix + "_Run" + str(CBF_RRT_strr_obj.iRun)
    else:
        fileName = prefix + "_iter" + str(CBF_RRT_strr_obj.i) + "_adaIiter" + str(adapDist_iter) + "_" + suffix + "_Run" + str(CBF_RRT_strr_obj.iRun)

    fileFullName = results_dir + fileName
    if enFlag:
        outfile = open(fileFullName, 'wb')
        if not isinstance(dataAsList,list):
            dataAsList = [dataAsList]
        pickle.dump([dataAsList], outfile)
        outfile.close()
def arcLength(qTrajectory):

    x = qTrajectory[0,:]
    y = qTrajectory[1,:]
    arc_Length = 0

    for i in range(len(qTrajectory[0,:])):
        if i==0:
            arc_Length = 0
        else:
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            darcLength = math.sqrt(dx**2+dy**2)
            arc_Length = arc_Length + darcLength

    return arc_Length

def si_to_uni_dyn(dxi, poses):
    """A mapping from single-integrator to unicycle dynamics.
        dxi: 2xN numpy array with single-integrator control inputs
        poses: 2xN numpy array with single-integrator poses
        -> 2xN numpy array of unicycle control inputs
        """

    #Check user input types
    assert isinstance(dxi, np.ndarray), "In the si_to_uni_dyn function created by the create_si_to_uni_dynamics function, the single integrator velocity inputs (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
    assert isinstance(poses, np.ndarray), "In the si_to_uni_dyn function created by the create_si_to_uni_dynamics function, the current robot poses (poses) must be a numpy array. Recieved type %r." % type(poses).__name__

    #Check user input ranges/sizes
    assert dxi.shape[0] == 2, "In the si_to_uni_dyn function created by the create_si_to_uni_dynamics function, the dimension of the single integrator velocity inputs (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
    assert poses.shape[0] == 3, "In the si_to_uni_dyn function created by the create_si_to_uni_dynamics function, the dimension of the current pose of each robot must be 3 ([x;y;theta]). Recieved dimension %r." % poses.shape[0]
    assert dxi.shape[1] == poses.shape[1], "In the si_to_uni_dyn function created by the create_si_to_uni_dynamics function, the number of single integrator velocity inputs must be equal to the number of current robot poses. Recieved a single integrator velocity input array of size %r x %r and current pose array of size %r x %r." % (dxi.shape[0], dxi.shape[1], poses.shape[0], poses.shape[1])
    linear_velocity_gain = 1 
    angular_velocity_limit = np.pi

    M,N = np.shape(dxi)

    a = np.cos(poses[2, :])
    b = np.sin(poses[2, :])

    dxu = np.zeros((2, N))
    dxu[0, :] = linear_velocity_gain*(a*dxi[0, :] + b*dxi[1, :])
    dxu[1, :] = angular_velocity_limit*np.arctan2(-b*dxi[0, :] + a*dxi[1, :], dxu[0, :])/(np.pi/2)

    return dxu


def isAcceptableSample(traj,desiredSteps_num):
    """
    Check if the number of the steps of the trajectory is the same as the number of the steps that are fed to the safeSteerr method.
    :param traj: The state trajectory generated through the safeSteer method
    :param desiredSteps_num: the number of steps that is fed to the safeSteer method
    :return: True if the number of the steps in of the trajectory are the same as the desiredSteps_num.
    """
    # return int(len(traj[0,:])-1)==int(desiredSteps_num)
    return (int(len(traj)) <= int(desiredSteps_num+1)) and (int(len(traj)) >= int(desiredSteps_num-2))
def main(worldChar='RSS_paper'):
    Params('configs.ini')

    a = list(np.linspace(29, 49, 21))
    for j in [0]:
        for iRun in [10]:
            try:
                runSeed = int(iRun+1)
                np.random.seed(runSeed)
                random.seed(runSeed)
                sys.setrecursionlimit(2000)
                # worldChar = 'Cltrd_world_big'
                if worldChar is 'Cltrd_world':
                    q_init = np.array([.8, 2.5, 0])
                    xy_goal = np.array([5.5, 4.5])
                elif worldChar is 'Cltrd_world_big':
                    q_init = np.array([1, 2, 0])
                    xy_goal = np.array([8.5, 6.6])
                elif worldChar is 'Cltrd_world_big2':
                    q_init = np.array([1, 4, 0])
                    xy_goal = np.array([10, 4])
                elif worldChar is 'nPsgs_world':
                    q_init = np.array([1, 2, 0])
                    xy_goal = np.array([8.5, 6.8])
                elif worldChar is 'Smpl_world':
                    q_init = np.array([0, 0, 0])
                    xy_goal = np.array([24., 16.])
                elif worldChar is 'NoObs_world':
                    q_init = np.array([1, 2, 0])
                    xy_goal = np.array([8.5, 6.5])
                elif worldChar is 'Circ_world':
                    q_init = np.array([2, 12.5, 0])
                    xy_goal = np.array([23, 12.5])
                elif worldChar is 'CircCross_world':
                    q_init = np.array([2, 12.5, 0])
                    xy_goal = np.array([23, 12.5])
                elif worldChar is 'RSS_paper':
                    q_init = np.array([0.5, 3, 0])
                    xy_goal = np.array([15, 13])

                #File naming stuff:

                CBF_RRT_object = CBF_RRTstrr(suffix=worldChar,q_init=q_init,xy_goal=xy_goal,eps_g=.5)
                # if j == 0:
                #     CBF_RRT_object.kde_enabled = False
                # else:
                #     CBF_RRT_object.kde_enabled = False

                if CBF_RRT_object.kde_enabled or CBF_RRT_object.params.CE_enabled:
                    prefix = 'adap'
                elif CBF_RRT_object.CBF_RRT_enable:
                    prefix = 'RRT_'
                else:
                    prefix = '_'
                CBF_RRT_object.prefix = prefix
                CBF_RRT_object.Initialization(worldChar=worldChar)
                # CBF_RRT_object.initialize_graphPlot()
                # plt.show()
                CBF_RRT_object.iRun = int(iRun+1)
                #---- Manual plan:
                #----


                # plt.show()
                CBF_RRT_object.Plan_CBF_RRT_strr()
            except:
                raise
                print("Run:",iRun+1," has faild")


if __name__ == '__main__':
    start_time = time.time()
    worldChars = ['RSS_paper']# ['Cltrd_world_big', 'nPsgs_world']
    for worldChar in worldChars:
        main(worldChar)
    end_time = time.time()
    print("Total runtime:", end_time - start_time)
    
    
    #---------------------------------------------
    # The follwing part is for profiling the code:
    #---------------------------------------------
    # import cProfile 
    # cProfile.run('main()', "output3.dat")
    
    # import pstats
    # # from pstats import SortKey
    
    # # Defined based on the time:
    # with open("output_time2.txt", "w") as f:
    #     p = pstats.Stats("output3.dat", stream=f)
    #     p.sort_stats("time").print_stats()
    
    # # Defined based on the calls:
    # with open("output_calls2.txt", "w") as f:
    #     p = pstats.Stats("output3.dat", stream=f)
    #     p.sort_stats("calls").print_stats()