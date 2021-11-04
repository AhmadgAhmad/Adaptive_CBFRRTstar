import sys
print(sys.version)
import time
import numpy as np
import matplotlib.pyplot as plt
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
#from hocbf_dummy import HOCBF
from hocbf import *
from simulation import Simulation
import copy
import random
from sklearn.neighbors import NearestNeighbors
from expansionTree import ExpansionTree



class Vertex(object):
    def __init__(self, State = None, StateTraj = None, CtrlTraj = None, ParentIndexID = None, ChildrenIndexIDs = None,PathToCome = None, timeTraj =None,indexID=0,curTime=0,CostToGo = float('inf'),vgFlag=False):
        self.params          = Params()
       #Index/es and identification/s attributes:
        self.ParentIndexID = ParentIndexID #When add a parent index update the pathVertices as well as the CostToCome
        self.ChildrenIndexIDs = ChildrenIndexIDs
        self.indexID = indexID

        #---Trajectory attributes:
        self.StateTraj = StateTraj
        self.CtrlTraj = CtrlTraj
        self.timeTraj = timeTraj
        #self.PathToCome = self.updatePathToCome(TreeT)

        #---Current attributes:
        self.curTime = curTime
        self.State = State


        #---The costs attributes:
        self.CostToCome = None #TODO: Must not compute it everey time we call for the CostToCome (unless if we changed the parent)
        self.CostToGo = CostToGo
        self.PathToCome = PathToCome
        #---
        self.vgFlag = vgFlag
    ###############################  Methods: #################################################
    '''
    This method will get the costToCome of each node in the path and sum up the cost till it reaches 
    the current node (). Warning: this method will return an error if one attempt to compute the CostToCome to 
    disconnected node. 
    '''
    def computeCostToCome(self,TreeT):
        #Check if the vertex is the root vertex (has no parent); then initialize with a zero CostToCome
        ParentIndexID = self.ParentIndexID
        v_indexID = self.indexID
        if ParentIndexID is None and v_indexID !=0:
            #raise Exception('Attempted to compute the CostToCome to DISCONNECTED vertex (has no parent)')
            CostToCome = None
        elif v_indexID is 0:
            CostToCome = 0.0
        else:
            #--- Extracting the parent vertex from the tree:
            v_parent            = TreeT.VerticesSet[ParentIndexID]
            CostToCome_v_parent = v_parent.CostToCome
            Cost_v_parent2v     = self.Cost_v2v(v_start=v_parent)
            CostToCome          = CostToCome_v_parent + Cost_v_parent2v
        self.CostToCome=CostToCome


    def Cost_v2v(self,v_start):
        xy_v_start = v_start.State[0:2]
        xy_v_end   = self.State[0:2]
        EucNorm = np.linalg.norm(xy_v_end-xy_v_start)
        return EucNorm

    def updateParent(self,parentVertex):
        #The PathToCome is not necessary at all.

        # Update the PathToCome:
        # if self.PathToCome is None:
        #     self.PathToCome = parentVertex.PathToCome + [self.indexID]
        # else:
        #     #Truncate the PathToCome of the previous parent from the PathToCome to prepare for updating hte PathToCome with the new parent
        #     prvPrtVertex_PathToCome = parentVertex.PathToCome
        #     self.PathToCome = parentVertex.PathToCome + [self.indexID]
        #Update the ParentIndexID:
        self.ParentIndexID = parentVertex.indexID



    def addChild(self,chiledIndexID): #TODO: write Child correctly
        """
        Append the childIndexID to the ChildrenIndexIDs of the vertex (self)
        :param chiledIndexID:
        :return: No returns
        """
        curr_chiledIndexIDs = self.ChildrenIndexIDs
        if curr_chiledIndexIDs is None:
            curr_chiledIndexIDs = [chiledIndexID]
        else:
            curr_chiledIndexIDs.append(chiledIndexID)
        self.ChildrenIndexIDs = curr_chiledIndexIDs

    '''
    This method will update the vertices of the PathToCome to this vertex by retracing the parent's PathToCome 
    (this will be very handy afterwards with the CE framework!)
    '''
    # def updatePathToCome(self,TreeT):
    #     ParentIndexID = self.ParentIndexID
    #     v_indexID = self.indexID
    #     if ParentIndexID is None and v_indexID !=0:
    #         # raise Exception('Attempted to compute the CostToCome to DISCONNECTED vertex (has no parent)')
    #         PathToCome = None
    #     elif v_indexID is 0:
    #         PathToCome = 0.0
    #     else:
    #         # --- Extracting the parent vertex from the tree:
    #         v_parent = TreeT.VerticesSet[ParentIndexID]
    #         PathToCome_v_parent = v_parent.PathToCome
    #         PathToCome = np.append(PathToCome_v_parent,v_indexID)
    #     return PathToCome


    #
    #
    # def get_initTime(self):
    #     # Get the final time of its parent trajectory (when rewiring make sure that you updated the parent before updatingthe initial time)
    #     pass
    # def get_FinalTime(self):
    #     # After steering to the point; make sure to shift the final time trajectory that is generated by the safe steering
    #     pass
    # def get_initState(self):
    #     #Which will be the final state of the parent vertex
    #     pass
    # def get_finalState(self):
    #     #
    #     pass
    # def cat_timeTraj(self):
    #     #Concatinate the time trajectory
    #     pass

def main():
    Params('configs.ini')
    v_1Traj = np.vstack([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    v_2Traj = 2 * np.vstack([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    TreeT = ExpansionTree()
    v_0 = Vertex(State=np.array([0., 0., 9.]),indexID=0,TreeT=TreeT)
    TreeT.addVertex(v_0)
    v_1 = Vertex(State=np.array([0.5, 0., 9.]), StateTraj=v_1Traj, CtrlTraj=.1 * v_1Traj, timeTraj=.01 * v_1Traj,
                 indexID=1,ParentIndexID=0,TreeT=TreeT)
    TreeT.addVertex(v_1)
    v_2 = Vertex(State=np.array([1., 0., 18.]), StateTraj=v_2Traj, CtrlTraj=.1 * v_2Traj, timeTraj=.01 * v_2Traj,
                 indexID=2,ParentIndexID=1,TreeT=TreeT)
    TreeT.addVertex(v_2)
    v_3 = Vertex(State=np.array([2., 0., 18.]), StateTraj=v_2Traj, CtrlTraj=.1 * v_2Traj, timeTraj=.01 * v_2Traj,
                 indexID=3,ParentIndexID=2,TreeT=TreeT)
    TreeT.addVertex(v_3)
    v_4 = Vertex(State=np.array([3., 0., 18.]), StateTraj=v_2Traj, CtrlTraj=.1 * v_2Traj, timeTraj=.01 * v_2Traj,
                 indexID=4,ParentIndexID=3,TreeT=TreeT)
    TreeT.addVertex(v_4)
    v_5 = Vertex(State=np.array([4., 0., 9.]), StateTraj=v_1Traj, CtrlTraj=.1 * v_1Traj, timeTraj=.01 * v_1Traj,
                 indexID=5,ParentIndexID=3,TreeT=TreeT)
    TreeT.addVertex(v_5)
    v_1.Cost_v2v(v_2)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime:", end_time - start_time)