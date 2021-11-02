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
from collections import OrderedDict
#from vertex import Vertex

class ExpansionTree(object):
    def __init__(self,VerticesSet = None, EdgesSet = None,xyCoorVertecis = None):
        self.VerticesSet = VerticesSet
        self.VSet_forNNs = None
        self.EdgesSet = EdgesSet
        # self.xyCoorVertices = []
        self.xyCoorVertices = xyCoorVertecis
        self.xyVset_forNNs = None
        self.params = Params()

    def addVertex(self,VertexToAdd):
        #Update the Vertices Set
        v_indexID = VertexToAdd.indexID
        currVerticesSet = self.VerticesSet
        currVSet_forNNs = self.VSet_forNNs
        if self.params.ordDict_enabled:
            if currVerticesSet is None:
                currVerticesSet = OrderedDict()
            currVerticesSet.update({v_indexID:VertexToAdd}) #TODO (slow)
            toReturnVerticesSet = currVerticesSet   #TODO(slow)
            self.VerticesSet = toReturnVerticesSet

            if currVSet_forNNs is None:
                currVSet_forNNs = []
            currVSet_forNNs.append(VertexToAdd)
            self.VSet_forNNs = currVSet_forNNs

            #Update the xyCoorVerticesDict
            curr_xyCoorVertices = self.xyCoorVertices
            if curr_xyCoorVertices is None:
                curr_xyCoorVertices = OrderedDict()
            curr_xyCoorVertices.update({v_indexID:VertexToAdd.State[0:2]})
            Return_xyCoorVertices  =curr_xyCoorVertices
            self.xyCoorVertices = Return_xyCoorVertices

            curr_xyVset_forNNs = self.xyVset_forNNs
            if curr_xyVset_forNNs is None:
                curr_xyVset_forNNs = []
            curr_xyVset_forNNs.append(VertexToAdd.State[0:2])
            Return_xyVset_forNNs  =curr_xyVset_forNNs
            self.xyVset_forNNs = Return_xyVset_forNNs

        else:
            if currVerticesSet is None:
                currVerticesSet = []
            currVerticesSet.append(VertexToAdd)
            self.VerticesSet = currVerticesSet

            # Update the xyCoorVerticesDict
            curr_xyCoorVertices = self.xyCoorVertices
            if curr_xyCoorVertices is None:
                curr_xyCoorVertices = []
            curr_xyCoorVertices.append(VertexToAdd.State[0:2])
            self.xyCoorVertices = curr_xyCoorVertices


        #Given that the tree has been updated, computeCostToCome to the added vertex to the tree if it has a parent:
        #TODO (caution): updateParentId and double computation of the cost to come

    def updateVertex(self,VertexToUpdate):
        """This method will update the vertex based on its indexID in the tree. We update the xyCoor explicitly for less
        complexity when we need the coordinates of the vertices in the tree"""

        #Update the ordered dict database (for tracking the when choosing parent and rewiring)
        vertexID = VertexToUpdate.indexID
        self.VerticesSet[vertexID]    = VertexToUpdate
        self.xyCoorVertices[vertexID] = VertexToUpdate.State[0:2]

        #Update the list database (for faster implemntation of Near and Nearest):
        self.VSet_forNNs[vertexID] = VertexToUpdate
        self.xyVset_forNNs[vertexID] = VertexToUpdate.State[0:2]

        # curr_xyCoorVertices = self.xyCoorVertices
        # if len(curr_xyCoorVertices) == 0:
        #     Return_xyCoorVertices = [VertexToUpdate.State[0:2]]
        # else:
        #     Return_xyCoorVertices = np.append(curr_xyCoorVertices, [VertexToUpdate.State[0:2]], axis=0)
        # self.xyCoorVertices = Return_xyCoorVertices


    def addEdge(self,iParent,iChild):
        currEdgesSet = self.EdgesSet
        Edge = np.array([iParent,iChild])
        if currEdgesSet is None:
            toReturnEdgesSet = [Edge]
        else:
            currEdgesSet.append(Edge)
            toReturnEdgesSet = currEdgesSet

        self.EdgesSet = toReturnEdgesSet

if __name__ == '__main__':
    tree = ExpansionTree()

    a= 2