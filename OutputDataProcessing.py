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
import scipy.stats as st
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression


def CostFile_oData(fileName):
    """

    :param fileName: the full name of the file of the data of the cost od the trajectory with the minimum cost
    that reach the goal.
    :return: oData_list: The list of the cost of the trajectories.
             itersV: The vector of the iterations of the vertexes that has been used o reach the goal.
    """

    #Get the directory of the output files:
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'OutputData/')
    figresults_dir = os.path.join(script_dir, 'Figures/')

    full_fileName = results_dir + fileName
    infile = open(full_fileName, 'rb')
    outputData = pickle.load(infile)
    costToCome_list = outputData[0][1]
    initIter = outputData[0][0]
    infile.close()
    if initIter is None:
        a = 1
    itersV = np.linspace(initIter, len(costToCome_list)+initIter, len(costToCome_list))
    itersV = itersV.astype(int)

    return costToCome_list, itersV



def TimeFile_oData(fileName):
    """

    :param fileName: the full name of the file of the data of the cost od the trajectory with the minimum cost
    that reach the goal.
    :return: oData_list: The list of the cost of the trajectories.
             iGoal: the first iteration when the goal has been reached.
    """

    # Get the directory of the output files:
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'OutputData/')
    figresults_dir = os.path.join(script_dir, 'Figures/')

    full_fileName = results_dir + fileName
    infile = open(full_fileName, 'rb')
    outputData = pickle.load(infile)
    timeIter_list = outputData[0]
    infile.close()

    #Accmulate the time of iterations:
    timePrev = 0
    accTime_list = []

    for i in range(len(timeIter_list)):
        time_i = timeIter_list[i] + timePrev
        timePrev = time_i
        accTime_list.append(time_i)

    #Use RBF support vector regression to fit a curve on the time of each iteration
    svr_rbf = KernelRidge(alpha=.1)
    ir = IsotonicRegression(out_of_bounds="clip")
    xPoints = np.linspace(0,len(timeIter_list),len(timeIter_list))
    timeIter_list = np.array(timeIter_list)
    timeIter_list_pred = ir.fit(xPoints, timeIter_list).predict(xPoints)

    # plt.scatter(xPoints, timeIter_list, color='green')
    # plt.plot(xPoints,timeIter_list_pred,color='red')
    # plt.show()

    return timeIter_list_pred, accTime_list

def TreeFile_oData(fileName):
    # Get the directory of the output files:
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'OutputData/')
    figresults_dir = os.path.join(script_dir, 'Figures/')

    full_fileName = results_dir + fileName
    infile = open(full_fileName, 'rb')
    outputData = pickle.load(infile)
    tree_list = outputData
    infile.close()

    return  tree_list

def KDEdistFile_oData(fileName):
    # Get the directory of the output files:
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'OutputData/')
    figresults_dir = os.path.join(script_dir, 'Figures/')

    gridFullFileName = results_dir + fileName
    infile = open(gridFullFileName, 'rb')
    KDEgrid_list = pickle.load(infile)

    return KDEgrid_list

def main(worldChar,plotOption=True,costFlag=True,timeFlag=True,treeFlag=True,distFlag=True,cprPlot=True):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'OutputData/')
    figresults_dir = os.path.join(script_dir, 'Figures/')
    Params('configs.ini')
    np.random.seed(2)
    random.seed(2)
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

    # File naming stuff:
    CBF_RRT_object = CBF_RRTstrr(suffix=worldChar, q_init=q_init, xy_goal=xy_goal, eps_g=.5)
    CBF_RRT_object.Initialization(worldChar=worldChar)
    if CBF_RRT_object.params.kde_enabled or CBF_RRT_object.params.kde_enabled:
        prefix = 'adap'
    else:
        prefix = '_'
    CBF_RRT_object.prefix = prefix
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>> Final results (all the examples are solely for 100 iterations) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #Plot Latex setting:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    #
    ###Plotting the cost stuff:
    if costFlag:
        itersV_list = []
        runCost_list = []
        baseNames = ['_CBF_RRTstr_Cost_iter2000_'+ worldChar + '_Run']#['adapCBF_RRTstr_Cost_iter400_'+ worldChar + '_Run','_CBF_RRTstr_Cost_iter400_'+ worldChar + '_Run']#,'_CBF_RRTstr_Cost_iter2000_Cltrd_world_big_Run','adapCBF_RRTstr_Cost_iter2000_Cltrd_world_big_Run']
        iName = 1
        # baseNames = ['adapCBF_RRTstr_Cost_iter2000_nPsgs_world_Run','_CBF_RRTstr_Cost_iter2000_nPsgs_world_Run']
        for baseName in baseNames:
            figName = baseName + ".pdf"
            for i in  range(20):
                if  i==2 or i==19:# or i==59 or i==61 or i==70 or i==72 or i==73 or i==78 or i==81 or i==94:
                      continue
                fileName = baseName+str(i+1)
                run_i_Cost,itersV = CostFile_oData(fileName)
                itersV_list.append(itersV)
                runCost_list.append(run_i_Cost)
                nData = len(run_i_Cost)
                # plt.plot(np.linspace(itersV,100,nData),run_i_Cost)

            #Computing the confidence intervals:
            upperC = []
            lowerC =[]
            mean = []
            for i in range(2000):
                # iter_i_list = [itersVt[i] for itersVt in itersV_list if itersVt[i]==i+1]
                indexes_i = []
                iter_i_runCost_list = []
                if i == 9:
                    b = 1
                for j in range(len(itersV_list)):
                    itersVt = list(itersV_list[j])
                    run_i_Cost = runCost_list[j]
                    if j == 5:
                        a = 1
                    try:
                        index = itersVt.index(i+1)
                    except:
                        index = None
                    indexes_i.append(index)
                    if index is not None:
                        if index>=len(run_i_Cost):
                            index = -1
                        iter_i_runCost_list.append(run_i_Cost[index])
                #Having iter_i_run_i_Cost compute its confidence bound and the mean:
                data = iter_i_runCost_list
                if len(data) ==0:
                    lowerC.append(None)
                    upperC.append(None)
                    mean.append(None)
                elif len(data)==1:
                    lowerC.append(data[0])
                    upperC.append(data[0])
                    mean.append(data[0])
                elif len(data)>1:
                    lc,uc = st.t.interval(alpha=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))
                    meanT = (lc+uc)/2
                    if abs(lc-meanT)>1:
                        lc= meanT
                        uc = meanT
                    lowerC.append(lc)
                    upperC.append(uc)
                    mean.append(meanT)

            olen = len(mean)
            lowerCt = [lc for lc in lowerC if lc!=None]
            upperCt = [uc for uc in upperC if uc != None]
            meant = [mn for mn in mean if mn != None]
            initIter = olen - len(meant)
            itersV = np.linspace(initIter, len(meant) + initIter, len(meant))
            if iName is 1:
                color ='blue'
            else:
                color = 'green'
            plt.plot(itersV,meant)
            plt.fill_between(itersV,lowerCt,upperCt, alpha=0.25,color= color)
            #Plot labeling and setting:
            plt.legend(['CBF-RRT$^*$'])#'Adaptive CBF-RRT$^*$',
            plt.xlabel(r'Number of vertexes', fontsize=16)
            plt.ylabel(r"Path length", fontsize=16)
            plt.xticks(size=14)
            plt.yticks(size=14)
            iName = iName + 1
        figFullName = figresults_dir + figName
        plt.grid(True)
        plt.savefig(figFullName)
        plt.show()
    ###################################################################################################################
    #Plotting the accmulative time stuff:
    if timeFlag:
        baseNames = ['_CBF_RRTstr_iterTime_iter2000_'+ worldChar + '_Run']#['adapCBF_RRTstr_iterTime_iter400_'+ worldChar + '_Run','_CBF_RRTstr_iterTime_iter400_'+ worldChar + '_Run']#,'_CBF_RRT_iterTime_iter2000_Cltrd_world_big_Run']
        iName = 1
        for iT_type in range(2):
            iName = 1
            for baseName in baseNames:
                runTime_list = []
                accTime_list = []
                for i in range(20):

                    fileName = baseName+str(i+1)
                    run_i_time, acc_i_time =  TimeFile_oData(fileName)

                    runTime_list.append(run_i_time)
                    accTime_list.append(acc_i_time)


                #Computing the confidence intervals:
                upperC = []
                lowerC =[]
                mean = []
                for i in range(1998):
                    indexes_i = []
                    iter_i_runTime_list = []
                    for j in range(20):
                        if iT_type is 0:
                            iter_i_runTime_run_j = runTime_list[j][i]
                            figName = 'v_runTime' + str(i) + "_" + worldChar + ".pdf"
                        else:
                            iter_i_runTime_run_j = accTime_list[j][i]
                            figName = 'v_accrunTime' + str(i) + "_" + worldChar + ".pdf"
                        iter_i_runTime_list.append(iter_i_runTime_run_j)
                    #Having iter_i_run_i_Cost compute its confidence bound and the mean:
                    data = iter_i_runTime_list
                    lc,uc = st.t.interval(alpha=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))
                    lowerC.append(lc)
                    upperC.append(uc)
                    mean.append((lc+uc)/2)
                mean = mean
                lowerC = lowerC
                upperC = upperC
                if iName is 1:
                    color = 'blue'
                else:
                    color = 'green'
                plt.plot(np.linspace(0,len(mean),len(mean)),mean)
                plt.fill_between(np.linspace(0,len(mean),len(mean)),lowerC,upperC, alpha=0.25,color= color)
                #Plot labeling and setting:
                plt.legend(['CBF-RRT$^*$'],fontsize=16)#'Adaptive CBF-RRT$^*$',
                plt.xlabel(r'Number of vertexes', fontsize=16)
                plt.ylabel(r'CPU time $[s]$', fontsize=16)
                plt.xticks(size=14)
                plt.yticks(size=14)
                iName = iName+1

            figFullName = figresults_dir + figName
            plt.grid(True)
            plt.savefig(figFullName)
            plt.show()

    ###################################################################################################################
    # Plotting the tree and paths stuff:
    if treeFlag:
        fileName = '_CBF_RRTstr_Tree_iter2000_Cltrd_world_big_Run8'
        figName = fileName +'.pdf'
        Tree_asList  = TreeFile_oData(fileName)
        CBF_RRT_object.TreeT = Tree_asList[0][0]
        CBF_RRT_object.initialize_graphPlot()
        goalVertex = CBF_RRT_object.plot_tree(Tree_asList[0][1][-1], plot_pathFalg=True)
        figFullName = figresults_dir + figName
        plt.savefig(figFullName)
        # CBF_RRT_object.TreePlot.show()

    ###################################################################################################################
    # Plotting the distribution stuff:
    if distFlag:
        # Get the grid of the distribution as well as the elite set:
        fileName  = 'adapCBF_RRTstr_KDEgridProbs_iter372_adaIiter3_nPsgs_world_Run8'
        figFullName = figresults_dir+fileName+'.pdf'
        KDEgrid_list = KDEdistFile_oData(fileName)
        Xxgrid = KDEgrid_list[0][0]
        Xygrid = KDEgrid_list[0][1]
        gridProbs = KDEgrid_list[0][2]
        elite_samples_arr = KDEgrid_list[0][3]
        CBF_RRT_object.initialize_graphPlot()
        # The distribution levelsets:
        CS = plt.contour(Xxgrid, Xygrid, gridProbs)  # , norm=LogNorm(vmin=4.18, vmax=267.1))
        plt.colorbar(CS, shrink=0.8, extend='both')
        plt.scatter(elite_samples_arr[:, 0], elite_samples_arr[:, 1])
        plt.savefig(figFullName)
        # plt.show()
        a = 1

    if cprPlot:
        fileNames = ['CBF_CLF_PathToGCltrd_world_big2','adapCBF_RRTstr_PathToG_iter1000_Cltrd_world_big2_Run8','_CBF_RRTstr_PathToG_iter1000_Cltrd_world_big2_Run8']
        figName =  'CBF_CLF_RRT_cltrd2.pdf'
        CBF_RRT_object.initialize_graphPlot()
        for fileName in fileNames:
            pathToG = KDEdistFile_oData(fileName)
            pathToG = pathToG[0][0]
            plt.plot(pathToG[0,:],pathToG[1,:],markersize=20)
        figFullName = figresults_dir + figName
        plt.legend(['CBF-CLF QP', 'Adaptive CBF-RRT$^*$','CBF-RRT$^*$'], fontsize=14)
        # plt.savefig(figFullName)
        plt.show()


    # >>>>>>>>>>>>>>>>>> =================<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #-------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    worldChars = ['Cltrd_world_big']#['Cltrd_world_big','nPsgs_world']#Cltrd_world_big'
    plotOption = 'cost_&_tree'#'cost_&_tree' # 'dist_&_tree'
    for worldChar in worldChars:
        main(worldChar,costFlag=False,timeFlag=False,treeFlag=True,distFlag=False,cprPlot=False)