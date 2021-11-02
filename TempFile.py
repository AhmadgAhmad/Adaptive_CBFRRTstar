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
from CBF_RRT_strr import *
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


def main(worldChar,plotOption):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'OutputData/')
    figresults_dir = os.path.join(script_dir, 'Figures/')
    Params('configs.ini')
    np.random.seed(2)
    random.seed(2)
    sys.setrecursionlimit(2000)

    # worldChar = 'Cltrd_world_big'
    if worldChar is 'Cltrd_world':
        q_init = np.array([1, 2, 0])
        xy_goal = np.array([5.5, 4.5])
    elif worldChar is 'Cltrd_world_big':
        q_init = np.array([1, 2, 0])
        xy_goal = np.array([8.5, 6.5])
    elif worldChar is 'nPsgs_world':
        q_init = np.array([0, 0, 0])
        xy_goal = np.array([24., 16.])
    elif worldChar is 'Smpl_world':
        q_init = np.array([0, 0, 0])
        xy_goal = np.array([24., 16.])
    elif worldChar is 'NoObs_world':
        q_init = np.array([0, 0, 0])
        xy_goal = np.array([20., 16.])
    elif worldChar is 'Circ_world':
        q_init = np.array([2, 12.5, 0])
        xy_goal = np.array([23, 12.5])
    elif worldChar is 'CircCross_world':
        q_init = np.array([2, 12.5, 0])
        xy_goal = np.array([23, 12.5])

    # File naming stuff:
    CBF_RRT_object = CBF_RRTstrr(suffix=worldChar, q_init=q_init, xy_goal=xy_goal, eps_g=2)
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
    itersV_list = []
    runCost_list = []
    baseNames = ['adapCBF_RRTstr_Cost_iter2000_Cltrd_world_big_Run','_CBF_RRTstr_Cost_iter2000_Cltrd_world_big_Run']#,'_CBF_RRTstr_Cost_iter2000_Cltrd_world_big_Run','adapCBF_RRTstr_Cost_iter2000_Cltrd_world_big_Run']
    iName = 1
    # baseNames = ['adapCBF_RRTstr_Cost_iter2000_nPsgs_world_Run','_CBF_RRTstr_Cost_iter2000_nPsgs_world_Run']
    for baseName in baseNames:

        for i in  range(20):
            # if  i==23:# or i==26 or i==59 or i==61 or i==70 or i==72 or i==73 or i==78 or i==81 or i==94:
            #      continue
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
                lowerC.append(lc)
                upperC.append(uc)
                mean.append((lc+uc)/2)

        olen = len(mean)
        lowerCt = [lc for lc in lowerC if lc!=None]
        upperCt = [uc for uc in upperC if uc != None]
        meant = [mn for mn in mean if mn != None]
        initIter = olen - len(meant)
        itersV = np.linspace(initIter, len(meant) + initIter, len(meant))
        if iName is 1:
            color ='yellow'
        else:
            color = 'green'
        plt.plot(itersV[50:-3],meant[50:-3])
        plt.fill_between(itersV[50:-3],lowerCt[50:-3],upperCt[50:-3], alpha=0.2,color= color)
        #Plot labeling and setting:
        plt.legend(['Adaptive CBF-RRT$^*$','CBF-RRT$^*$'])
        plt.xlabel(r'Number of vertexes', fontsize=16)
        plt.ylabel(r"Path length", fontsize=16)
        plt.xticks(size=14)
        plt.yticks(size=14)
        iName = iName + 1
    plt.grid(True)
    plt.show()
    ###################################################################################################################
    #Plotting the accmulative time stuff:


    baseNames = ['adapCBF_RRTstr_iterTime_iter2000_Cltrd_world_big_Run','_CBF_RRTstr_iterTime_iter2000_Cltrd_world_big_Run']#,'_CBF_RRT_iterTime_iter2000_Cltrd_world_big_Run']
    iName = 1
    for iT_type in range(2):
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
            for i in range(1950):
                indexes_i = []
                iter_i_runTime_list = []
                for j in range(20):
                    if iT_type is 0:
                        iter_i_runTime_run_j = runTime_list[j][i]
                    else:
                        iter_i_runTime_run_j = accTime_list[j][i]

                    iter_i_runTime_list.append(iter_i_runTime_run_j)
                #Having iter_i_run_i_Cost compute its confidence bound and the mean:
                data = iter_i_runTime_list
                lc,uc = st.t.interval(alpha=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))
                lowerC.append(lc)
                upperC.append(uc)
                mean.append((lc+uc)/2)
            mean = mean[1:-9]
            lowerC = lowerC[1:-9]
            upperC = upperC[1:-9]

            # olen = len(mean)
            # lowerCt = [lc for lc in lowerC if lc!=None]
            # upperCt = [uc for uc in upperC if uc != None]
            # meant = [mn for mn in mean if mn != None]
            # initIter = olen - len(meant)
            # itersV = np.linspace(initIter, len(meant) + initIter, len(meant))
            if iName is 1:
                color = 'yellow'
            else:
                color = 'green'
            plt.plot(np.linspace(0,len(mean),len(mean)),mean)
            plt.fill_between(np.linspace(0,len(mean),len(mean)),lowerC,upperC, alpha=0.5,color= color)
            #Plot labeling and setting:
            plt.legend(['Adaptive CBF-RRT$^*$','CBF-RRT$^*$'])
            plt.xlabel(r'Number of vertexes', fontsize=16)
            plt.ylabel(r'CPU time $[s]$', fontsize=16)
            plt.xticks(size=14)
            plt.yticks(size=14)
            iName = iName+1
        figName = 'iterationTime' + str(i) + "_" + worldChar + ".pdf"
        figFullName = figresults_dir + figName
        plt.grid(True)
        plt.savefig(figFullName)
        plt.show()


    # >>>>>>>>>>>>>>>>>> =================<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #-------------------------------------------------------------------------------------------------------------------
    #Pltting:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    iter = [1000]#[500,1000,2500,5000,10000]
    if plotOption == 'cost_&_tree':
       for i in iter:
           AdapCBF_RRTstr_costFileName = 'CBF_RRTstr_iterTime_iter' + str(i) + "_" + worldChar
           AdapCBF_RRTstr_costFileFillName = results_dir + AdapCBF_RRTstr_costFileName
           infile = open(AdapCBF_RRTstr_costFileFillName, 'rb')
           goal_costToCome_list_AdapCBF_RRTstr = pickle.load(infile)
           goal_costToCome_list_AdapCBF_RRTstr = goal_costToCome_list_AdapCBF_RRTstr[0]
           infile.close()

           # Figuring out the number of initial number of vertexes:
           # num_init_CBF_RRTstar = i - len(goal_costToCome_list_CBF_RRTstr[1:-1])
           # cost_init_CBF_RRTstar = goal_costToCome_list_CBF_RRTstr[1]
           # # costV_init_CBF_RRTstar = list(itertools.repeat(cost_init_CBF_RRTstar, num_init_CBF_RRTstar))
           # num_init_adapCBF_RRTstar = i - len(goal_costToCome_list_AdapCBF_RRTstr[1:-1])
           # cost_init_adapCBF_RRTstar = goal_costToCome_list_AdapCBF_RRTstr[1]
           # costV_init_adapCBF_RRTstar = list(itertools.repeat(cost_init_adapCBF_RRTstar, num_init_adapCBF_RRTstar))
           # # goal_costToCome_list_CBF_RRTstr = goal_costToCome_list_CBF_RRTstr[1:-1]
           # goal_costToCome_list_AdapCBF_RRTstr = goal_costToCome_list_AdapCBF_RRTstr[1:-1]
           # # costV_init_CBF_RRTstar.extend(goal_costToCome_list_CBF_RRTstr)
           # costV_init_adapCBF_RRTstar.extend(goal_costToCome_list_AdapCBF_RRTstr)

           # plt.plot(np.linspace(1, len(goal_costToCome_list_CBF_RRTstr)+num_init_CBF_RRTstar,
           #                      len(goal_costToCome_list_CBF_RRTstr)+num_init_CBF_RRTstar),
           #                      costV_init_CBF_RRTstar,
           #                      label="Cost-to-come", hold='on', linewidth=2)
           plt.plot(list(np.linspace(1, len(goal_costToCome_list_AdapCBF_RRTstr),len(goal_costToCome_list_AdapCBF_RRTstr))),goal_costToCome_list_AdapCBF_RRTstr)
                    # ,
                    # label="Cost-to-come", hold='on', linewidth=2)

           figName = 'iterationTime' + str(i) + "_" + worldChar + ".pdf"
           figFullName = figresults_dir + figName
           plt.legend(['CBF-RRT$^*$', 'Adaptive CBF-RRT$^*$'])

           plt.xlabel(r'Number of vertexes', fontsize=16)
           plt.ylabel(r"$\texttt{costToCome}$", fontsize=16)
           plt.xticks(size=14)
           plt.yticks(size=14)
           plt.grid(True)
           plt.savefig(figFullName)
           plt.show()

           ##Time shit

            #The file of the adaptive CBF-RRT*
           AdapCBF_RRTstr_costFileName = 'adapCBF_RRTstr_Cost_iter'+str(i)+"_"+worldChar
           AdapCBF_RRTstr_costFileFillName = results_dir + AdapCBF_RRTstr_costFileName
           infile = open(AdapCBF_RRTstr_costFileFillName, 'rb')
           goal_costToCome_list_AdapCBF_RRTstr = pickle.load(infile)
           goal_costToCome_list_AdapCBF_RRTstr=goal_costToCome_list_AdapCBF_RRTstr[0][1]
           infile.close()

           #Figuring out the number of initial number of vertexes:
           # num_init_CBF_RRTstar = i - len(goal_costToCome_list_CBF_RRTstr[1:-1])
           # cost_init_CBF_RRTstar = goal_costToCome_list_CBF_RRTstr[1]
           # costV_init_CBF_RRTstar = list(itertools.repeat(cost_init_CBF_RRTstar, num_init_CBF_RRTstar))
           num_init_adapCBF_RRTstar = i - len(goal_costToCome_list_AdapCBF_RRTstr[1:-1])
           cost_init_adapCBF_RRTstar = goal_costToCome_list_AdapCBF_RRTstr[1]
           costV_init_adapCBF_RRTstar = list(itertools.repeat(cost_init_adapCBF_RRTstar, num_init_adapCBF_RRTstar))
           # goal_costToCome_list_CBF_RRTstr = goal_costToCome_list_CBF_RRTstr[1:-1]
           goal_costToCome_list_AdapCBF_RRTstr = goal_costToCome_list_AdapCBF_RRTstr[1:-1]
           # costV_init_CBF_RRTstar.extend(goal_costToCome_list_CBF_RRTstr)
           costV_init_adapCBF_RRTstar.extend(goal_costToCome_list_AdapCBF_RRTstr)

           # plt.plot(np.linspace(1, len(goal_costToCome_list_CBF_RRTstr)+num_init_CBF_RRTstar,
           #                      len(goal_costToCome_list_CBF_RRTstr)+num_init_CBF_RRTstar),
           #                      costV_init_CBF_RRTstar,
           #                      label="Cost-to-come", hold='on', linewidth=2)
           plt.plot(np.linspace(1, len(goal_costToCome_list_AdapCBF_RRTstr)+num_init_adapCBF_RRTstar,
                                len(goal_costToCome_list_AdapCBF_RRTstr)+num_init_adapCBF_RRTstar),
                                costV_init_adapCBF_RRTstar,
                                label="Cost-to-come", hold='on', linewidth=2)

           figName = 'CostToCome'+str(i)+"_"+worldChar+".pdf"
           figFullName = figresults_dir+figName
           plt.legend(['CBF-RRT$^*$', 'Adaptive CBF-RRT$^*$'])

           plt.xlabel(r'Number of vertexes', fontsize=16)
           plt.ylabel(r"$\texttt{costToCome}$", fontsize=16)
           plt.xticks(size=14)
           plt.yticks(size=14)
           plt.grid(True)
           plt.savefig(figFullName)
           plt.show()
           #Plot the corresponding the tree:

    elif plotOption == 'dist_&_tree':
        #Plot the trees of the predefined iterations
        iter = [ 500, 1000, 2500, 5000, 10000]
        for i in iter:
            # The file of CBF-RRT*
            CBF_RRTstr_costFileName = '_CBF_RRTstr_Tree_iter' + str(i) + "_" + worldChar
            CBF_RRTstr_costFileFillName = results_dir + CBF_RRTstr_costFileName
            infile = open(CBF_RRTstr_costFileFillName, 'rb')
            goal_Tree_list_CBF_RRTstr = pickle.load(infile)
            Tree_list_CBF_RRTstr = goal_Tree_list_CBF_RRTstr[0][0]
            vg_minCostToCome_CBF_RRTstr = goal_Tree_list_CBF_RRTstr[0][1][-1]
            infile.close()

            # The file of the adaptive CBF-RRT*
            AdapCBF_RRTstr_costFileName = 'adapCBF_RRTstr_Tree_iter' + str(i) + "_" + worldChar
            AdapCBF_RRTstr_costFileFillName = results_dir + AdapCBF_RRTstr_costFileName
            infile = open(AdapCBF_RRTstr_costFileFillName, 'rb')
            goal_Tree_list_AdapCBF_RRTstr = pickle.load(infile)
            Tree_list_AdapCBF_RRTstr = goal_Tree_list_AdapCBF_RRTstr[0][0]
            vg_minCostToCome_AdapCBF_RRTstr  = goal_Tree_list_AdapCBF_RRTstr[0][1][-1]
            infile.close()

            #Plot the tree of the CBF-RRT*
            CBF_RRT_object.TreeT = Tree_list_CBF_RRTstr
            CBF_RRT_object.initialize_graphPlot()
            goalVertex = CBF_RRT_object.plot_tree(vg_minCostToCome_CBF_RRTstr, plot_pathFalg=True)
            figName = 'Tree_CBF_RRTstr' + str(i) + "_" + worldChar + ".pdf"
            figFullName = figresults_dir + figName
            plt.savefig(figFullName)
            CBF_RRT_object.TreePlot.show()

            # Plot the tree of the CBF-RRT*
            CBF_RRT_object.TreeT = Tree_list_AdapCBF_RRTstr
            CBF_RRT_object.initialize_graphPlot()
            goalVertex = CBF_RRT_object.plot_tree(vg_minCostToCome_AdapCBF_RRTstr, plot_pathFalg=True)
            figName = 'Tree_AdapCBF_RRTstr' + str(i) + "_" + worldChar + ".pdf"
            figFullName = figresults_dir + figName
            plt.savefig(figFullName)
            CBF_RRT_object.TreePlot.show()
        #Plot the distribution with the corresponding tree as well as the elite samples:

    #Plot the distributions of cluttered world:
    iter = [216,306,363,430]
    j = 1
    for i in iter:
        #Get the tree and the goal vertex:
        TreeFileName = 'adapCBF_RRTstr_Tree_iter' + str(i) + "_"+'adaIiter'+str(j)+"_"+ worldChar
        TreeFullFileName = results_dir + TreeFileName
        infile = open(TreeFullFileName, 'rb')
        Tree_list = pickle.load(infile)
        Tree = Tree_list[0][0]
        vg = Tree_list[0][1][-1]
        infile.close()

        #Get the grid of the distribution as well as the elite set:
        gridFileName = 'adapCBF_RRTstr_KDEgridProbs_iter'+str(i)+'_adaIiter'+str(j)+'_'+worldChar
        gridFullFileName = results_dir + gridFileName
        infile = open(gridFullFileName, 'rb')
        KDEgrid_list = pickle.load(infile)
        Xxgrid = KDEgrid_list[0][0]
        Xygrid = KDEgrid_list[0][1]
        gridProbs = KDEgrid_list[0][2]
        elite_samples_arr = KDEgrid_list[0][3]

        #Plot the tree on top of the elite samples as well as the KDE:

        #The tree
        CBF_RRT_object.TreeT = Tree
        CBF_RRT_object.initialize_graphPlot()
        goalVertex = CBF_RRT_object.plot_tree(vg, plot_pathFalg=True)

        #The distribution levelsets:
        CS = plt.contour(Xxgrid, Xygrid, gridProbs)  # , norm=LogNorm(vmin=4.18, vmax=267.1))
        plt.colorbar(CS, shrink=0.8, extend='both')
        plt.scatter(elite_samples_arr[:, 0], elite_samples_arr[:, 1])

        figName = 'DistAdaptAndTree' + str(i) +'_adaIiter'+str(j)+ worldChar + ".pdf"
        figFullName = figresults_dir + figName
        plt.savefig(figFullName)
        CBF_RRT_object.TreePlot.show()
        j = j+1

    #============================================================================================
    #Animate the trajectory:
    fName = 'adapCBF_RRTstr_Tree_iter500_Cltrd_world'
    fFullName = results_dir + fName
    infile = open(fFullName, 'rb')
    goal_Tree_list = pickle.load(infile)
    Tree = goal_Tree_list[0][0]
    vg = goal_Tree_list[0][1][-1]
    infile.close()

    # Plot the tree of the CBF-RRT*
    CBF_RRT_object.TreeT = Tree
    CBF_RRT_object.initialize_graphPlot()
    goalVertex = CBF_RRT_object.plot_traj(vg, plot_pathFalg=True)
    figName = 'Tree_AdapCBF_RRTstr' + str(i) + "_" + worldChar + ".pdf"
    # figFullName = figresults_dir + figName
    # plt.savefig(figFullName)
    CBF_RRT_object.TrajPlot.show()





    # -------------------------------------------------------------------------------------------------------------------
    #Plotting the tree:
    TreeFileName = '_CBF_RRTstr_Tree_iter100_Cltrd_world'
    TreefileFullName = results_dir + TreeFileName
    infile = open(TreefileFullName, 'rb')
    Tree_asList = pickle.load(infile)
    infile.close()
    CBF_RRT_object.TreeT = Tree_asList[0][0]
    CBF_RRT_object.initialize_graphPlot()
    goalVertex = CBF_RRT_object.plot_tree(Tree_asList[0][1][-1], plot_pathFalg=True)
    CBF_RRT_object.TreePlot.show()


    #Plotting the cost:
    costFileName = '_CBF_RRTstr_Cost_iter100_Cltrd_world'
    costfileFullName = results_dir + costFileName
    infile = open(costfileFullName, 'rb')
    goal_costToCome_list = pickle.load(infile)
    infile.close()

    #Divide the list:
    costToCome_list = goal_costToCome_list[0][1]
    firstIter = goal_costToCome_list[0][0]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(np.linspace(firstIter, len(costToCome_list)+firstIter, len(costToCome_list)), costToCome_list,
             label="Cost-to-come", hold='on',linewidth=2)
    costFileName = 'adapCBF_RRTstr_Cost_iter100_Cltrd_world'
    costfileFullName = results_dir + costFileName
    infile = open(costfileFullName, 'rb')
    goal_costToCome_list = pickle.load(infile)
    infile.close()
    plt.plot(np.linspace(1, len(goal_costToCome_list[0]), len(goal_costToCome_list[0])), goal_costToCome_list[0],
             label="Cost-to-come", hold='on',linewidth=2)
    plt.legend(['Hi','there'])
    plt.xlabel(r'Number of vertexes',fontsize=16)
    plt.ylabel(r"$\texttt{costToCome}$", fontsize=16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(True)
    plt.savefig('line_plot.pdf')
    plt.show()





    #Plotting the distribution:
    distFileName = 'adapCBF_RRTstr_KDEgridProbs_iter48_adaIiter1_Cltrd_world'
    distfileFullName = results_dir + distFileName
    infile = open(distfileFullName, 'rb')
    distGrid_list = pickle.load(infile)
    infile.close()
    Xxgrid = distGrid_list[0][0]
    Xygrid = distGrid_list[0][1]
    grid_probs = distGrid_list[0][2]

    CBF_RRT_object.initialize_graphPlot()
    CS = plt.contour(Xxgrid, Xygrid, grid_probs.reshape(Xxgrid.shape))  # , norm=LogNorm(vmin=4.18, vmax=267.1))
    plt.colorbar(CS, shrink=0.8, extend='both')
    # plt.scatter(elite_samples_arr[:, 0], elite_samples_arr[:, 1])
    plt.show()
    a = 1


if __name__ == '__main__':
    worldChar = 'Cltrd_world_big'
    plotOption = 'cost_&_tree'#'cost_&_tree' # 'dist_&_tree'
    main(worldChar,plotOption)