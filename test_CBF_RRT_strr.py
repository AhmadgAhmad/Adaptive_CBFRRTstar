from CBF_RRT_strr import CBF_RRTstrr
from vertex import Vertex
import numpy as np
from expansionTree import ExpansionTree
from params import Params
from numpy.testing import *
import random
import math
import matplotlib.pyplot as plt






'''
------------------------------------------------------------------------------------------------------------------------
Testing the initialization procedure: 
------------------------------------------------------------------------------------------------------------------------
'''
def test_initialization_output_dimensions():
    assert True

def test_initialization_output_types():
    assert True

def test_initialization_output_Vg_leaves():
    assert True


'''
------------------------------------------------------------------------------------------------------------------------
Testing graph plotting methods
------------------------------------------------------------------------------------------------------------------------
'''
def test_initialize_graphPlot_output():

    assert True
'''
------------------------------------------------------------------------------------------------------------------------
Testing Functions for the Nearest Method: 
------------------------------------------------------------------------------------------------------------------------
'''
def test_Nearest_return_single_nearest_vertex_determenstic():
    """
    Test if Nearest return the nearest single vertex; and that it is the closest one.
    """
    #TODO --done : Run a randomized test
    #Setup:
    Params('configs.ini')
    """A set of vertices arranged as follows along the x-axis [0,1,2,3] and y-xis [0,1,2,3]: """
    v_1 = Vertex(State=np.array([0., 0., 9.]),indexID=1)
    v_2 = Vertex(State=np.array([1., 0., 9.]),indexID=2)
    v_3 = Vertex(State=np.array([2., 0., 9.]),indexID=3)
    v_4 = Vertex(State=np.array([3., 0., 9.]),indexID=4)
    v_5 = Vertex(State=np.array([0., 1., 9.]),indexID=5)
    v_6 = Vertex(State=np.array([0., 2., 9.]),indexID=6)
    v_7 = Vertex(State=np.array([0., 3., 9.]),indexID=7)
    v_8 = Vertex(State=np.array([1., 2., 9.]), indexID=8)
    v_9 = Vertex(State=np.array([2., 2., 9.]), indexID=9)
    v_10 = Vertex(State=np.array([3., 3., 9.]), indexID=10)

    """Create a Tree (given the tree is tested!!):"""
    tree = ExpansionTree()
    tree.addVertex(v_1)
    tree.addVertex(v_2)
    tree.addVertex(v_3)
    tree.addVertex(v_4)
    tree.addVertex(v_5)
    tree.addVertex(v_6)
    tree.addVertex(v_7)
    tree.addVertex(v_8)
    tree.addVertex(v_9)
    tree.addVertex(v_10)

    xy_sample = np.vstack([1,.5])
    CBF_RRT_object = CBF_RRTstrr()
    CBF_RRT_object.TreeT = tree

    #Exercise
    nearest_vertex = CBF_RRT_object.Nearest(xy_sample=xy_sample)

    #Verify:
    true_nearest_vertex = v_2

    assert nearest_vertex == true_nearest_vertex #Asseret the determenstic case

def test_Nearest_return_single_nearest_vertex_random():
    """Test if Nearest will return the nearest vertex of randomly sampled vertices as well as random testing point."""
    #Setup
    Params('configs.ini')
    tree = ExpansionTree()
    CBF_RRTstrr_object = CBF_RRTstrr()
    #Random sample:
    xy_sample = np.vstack([np.random.uniform(high=10),np.random.uniform(high=10)])
    NN_distance = 1000
    #Sample 10 Random Verteces:
    for i in range(10):
        State_vertex = np.array([np.random.uniform(high=10), np.random.uniform(high=10),1])
        xy_vertex    = State_vertex[0:2]
        v = Vertex(State=State_vertex,indexID=i)
        tree.addVertex(v)

        #Check for the actual nearest index
        norm_v2sample = np.linalg.norm(xy_vertex-xy_sample.ravel())
        if norm_v2sample < NN_distance:
            true_nn_indexID = i
            NN_distance = norm_v2sample

    CBF_RRTstrr_object.TreeT = tree
    #Exercise
    result_nnVertex = CBF_RRTstrr_object.Nearest(xy_sample)

    #Verify
    true_nnVertex = tree.VerticesSet[true_nn_indexID]

    #TODO I'm actually tricked here with the hashing thing
    assert result_nnVertex==true_nnVertex

'''
------------------------------------------------------------------------------------------------------------------------
Testing Functions for the Near Method: 
------------------------------------------------------------------------------------------------------------------------
'''
def test_Neare_return_near_vertecSet_determenstic():
    """
    Test if Nearest return the nearest single vertex; and that it is the closest one.
    """
    #TODO: Run a randomized test
    #Setup:
    Params('configs.ini')
    """A set of vertices arranged as follows along the x-axis [0,1,2,3] and y-xis [0,1,2,3]: """
    v_1 = Vertex(State=np.array([0., 0., 9.]),indexID=1)
    v_2 = Vertex(State=np.array([1., 0., 9.]),indexID=2)
    v_3 = Vertex(State=np.array([2., 0., 9.]),indexID=3)
    v_4 = Vertex(State=np.array([3., 0., 9.]),indexID=4)
    v_5 = Vertex(State=np.array([0., 1., 9.]),indexID=5)
    v_6 = Vertex(State=np.array([0., 2., 9.]),indexID=6)
    v_7 = Vertex(State=np.array([0., 3., 9.]),indexID=7)
    v_8 = Vertex(State=np.array([1., 2., 9.]), indexID=8)
    v_9 = Vertex(State=np.array([2., 2., 9.]), indexID=9)
    v_10 = Vertex(State=np.array([3., 3., 9.]), indexID=10)
    v_test = Vertex(State=np.array([.3, .45, 9.]), indexID=11)
    """Create a Tree (given the tree is tested!!):"""
    tree = ExpansionTree()
    tree.addVertex(v_1)
    tree.addVertex(v_2)
    tree.addVertex(v_3)
    tree.addVertex(v_4)
    tree.addVertex(v_5)
    tree.addVertex(v_6)
    tree.addVertex(v_7)
    tree.addVertex(v_8)
    tree.addVertex(v_9)
    tree.addVertex(v_10)
    tree.addVertex(v_test)

    CBF_RRT_object = CBF_RRTstrr()
    CBF_RRT_object.TreeT = tree
    #Exercise
    result_nearVerticesSet = CBF_RRT_object.Near(Vertex=v_test,b_raduis=1.5)

    #Verify:
    true_nearVerticesSet  = [v_1,v_5,v_2]
    assert result_nearVerticesSet == true_nearVerticesSet #Asseret the determenstic case


#TODO: Do the random test. The following test is a FAILING one.
#
# def test_Near_return_return_near_vertecSet_Random():
#     """Test if Nearest will return the nearest vertex of randomly sampled vertices as well as random testing point."""
#     #Setup
#     Params('configs.ini')
#     tree = ExpansionTree()
#     CBF_RRTstrr_object = CBF_RRTstrr()
#     #Random sample:
#     xy_sample = np.vstack([np.random.uniform(high=10),np.random.uniform(high=10)])
#     NN_distance = 100000
#     #Sample 10 Random Verteces:
#     for i in range(10):
#         State_vertex = np.array([np.random.uniform(high=10), np.random.uniform(high=10),1])
#         xy_vertex    = State_vertex[0:2]
#         v = Vertex(State=State_vertex,indexID=i)
#         tree.addVertex(v)
#
#         #Check for the actual nearest index
#         norm_v2sample = np.linalg.norm(xy_vertex-xy_sample.ravel())
#         if norm_v2sample < NN_distance:
#             true_nn_indexID = i
#             NN_distance = norm_v2sample
#
#
#     #Exercise
#     result_nnVertex = CBF_RRTstrr_object.Nearest(xy_sample)
#
#     #Verify
#     true_nnVertex = CBF_RRTstrr_object.TreeT.VerticesSet[true_nn_indexID]
#     assert result_nnVertex==true_nnVertex
#

'''
------------------------------------------------------------------------------------------------------------------------
Testing Functions for the SafeSteering Method: 
------------------------------------------------------------------------------------------------------------------------
'''

def test_SafeSteering_initial_final_times():
    """
    Test the initialTime and endTime of the trajectory generated using the SafeSteering method. The time of the beginning
    of the trajectory must be the endTime of the parent vertex. The endTime must be initilaTime + #steps*step_size
    """

    #Setup
    params = Params('configs.ini')
    nSteps = 15
    v_1Traj = np.vstack([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    v_1tTraj  = params.step_size * v_1Traj[0,:]
    v_parent = Vertex(State=np.array([0., 0., 9.]), StateTraj=v_1Traj, CtrlTraj=.1 * v_1Traj, timeTraj=v_1tTraj,curTime=v_1tTraj[-1],
                 indexID=1)
    CBF_RRTstrr_object = CBF_RRTstrr()

    #Exercise
    _, result_tFinal, _, _, result_tTrajectory = CBF_RRTstrr_object.SafeSteering(v_nearest=v_parent,desired_theta=0,m=nSteps)
    result_tInitial = result_tTrajectory[0]

    #Verify
    true_tInitial = v_1tTraj[-1]
    true_tFinal   = true_tInitial+nSteps*params.step_size

    assert_almost_equal(result_tInitial,true_tInitial)
    assert_almost_equal(result_tFinal,true_tFinal)

def test_SafeSteering_random_initial_final_pose():
    """
    Test steering from randomly generated initial and final pose.
    :return: (3 outputs)
    - True if path length is as expected.
    - True if initial final poses after steering are as expected.
    - True if the initial and final time as expected (based on the step_size as well as the path length)
    """
    params = Params('configs.ini')
    #Setup:

    #Randomly generate initial and final points and fine the norm to compute the number of steps of the safe steer
    xy_initial = np.vstack([random.uniform(-10,10),random.uniform(-10,10)])
    xy_final = np.vstack([random.uniform(-10, 10), random.uniform(-10, 10)])
    euc_norm = np.sqrt((xy_final.T-xy_initial.T).dot(xy_final-xy_initial))[0]
    step_size = params.step_size
    mSteps = euc_norm/step_size

    vertex_initial = Vertex(State=xy_initial)
    desired_theta = math.atan2(xy_final[1] - xy_initial[1], xy_final[0] - xy_initial[0])

    #Exercise:

    #SafeSteer to the point and generate the trajectories
    CBF_RRTstrr_object = CBF_RRTstrr()
    qFinal, tFinal, uTrajectory, qTrajectory, tTrajectory = CBF_RRTstrr_object.SafeSteering(vertex_initial, desired_theta, mSteps)

    result_pathLength = arcLength(qTrajectory)
    result_xy_initial = np.vstack([qTrajectory[0,0],qTrajectory[1,0]])
    result_xy_final   = np.vstack([qTrajectory[0,-1],qTrajectory[1,-1]])
    result_tFinal     = tFinal

    #Verify:
    true_pathLength = euc_norm[0]
    true_xy_initial = xy_initial
    true_xy_final   = xy_final
    true_tFinal     = mSteps*step_size

    assert_array_almost_equal(true_xy_initial,result_xy_initial)
    assert_array_almost_equal(true_xy_final,result_xy_final)
    assert_almost_equal(true_tFinal,result_tFinal)
    assert_almost_equal(true_pathLength,result_pathLength)

def test_SafeSteering_initial_final_state():
    """
    Test the initialState of the trajectory generated using the SafeSteering method. The state of the beginning
    of the trajectory must be the finalState of the parent vertex.
    """

    #Setup
    params = Params('configs.ini')
    nSteps = 15
    v_1Traj = np.vstack([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    v_1tTraj  = .1 * v_1Traj[0,:]
    v_parent = Vertex(State=np.array([3., 6., 9.]), StateTraj=v_1Traj, CtrlTraj=.1 * v_1Traj, timeTraj=v_1tTraj,curTime=v_1tTraj[-1],
                 indexID=1)
    CBF_RRTstrr_object = CBF_RRTstrr()

    #Exercise
    result_qFinal, _, _, result_qTrajectory,_ = CBF_RRTstrr_object.SafeSteering(v_nearest=v_parent,desired_theta=0,m=nSteps)
    result_qInitial = result_qTrajectory[:,0]

    #Verify
    true_qInitial = v_1Traj[:,-1]
    assert_array_almost_equal(result_qInitial[0:2],true_qInitial[0:2],decimal=1)

    #Clean-up

def test_SafeSteering_initial_final_state_with_obstacle():
    assert True

'''
------------------------------------------------------------------------------------------------------------------------
Testing Functions for the ChooseParent Method: 
------------------------------------------------------------------------------------------------------------------------
'''
def test_ChooseParent_output_determenstic():
    """Test the returned least expensive parent of the input vertex. The vertex and the near vertices are predetermined. """

    #Setup
    params = Params('configs.ini')
    """A set of vertices arranged as follows along the x-axis [0,1,2,3] and y-xis [0,1,2,3]: """

    v_0 = Vertex(State=np.array([0., 0., 9.]),StateTraj=np.array([0., 0., 9.]) ,indexID=0)
    v_1 = Vertex(State=np.array([1., 0., 9.]),StateTraj=np.array([1., 0., 9.]), indexID=1,ParentIndexID=0)
    v_2 = Vertex(State=np.array([2., 0., 9.]),StateTraj= np.array([2., 0., 9.]),indexID=2,ParentIndexID=1)
    v_3 = Vertex(State=np.array([3., 0., 9.]),StateTraj=np.array([3., 0., 9.]), indexID=3,ParentIndexID=2)
    v_4 = Vertex(State=np.array([0., 1., 9.]),StateTraj=np.array([0., 1., 9.]) ,indexID=4,ParentIndexID=0)
    v_5 = Vertex(State=np.array([0., 2., 9.]), StateTraj=np.array([0., 2., 9.]),indexID=5,ParentIndexID=4)
    v_6 = Vertex(State=np.array([0., 3., 9.]), StateTraj=np.array([0., 3., 9.]),indexID=6,ParentIndexID=5)
    # v_8 = Vertex(State=np.array([1., 2., 9.]), indexID=8)
    # v_9 = Vertex(State=np.array([2., 2., 9.]), indexID=9)
    # v_10 = Vertex(State=np.array([3., 3., 9.]), indexID=10)
    v_sample = Vertex(State=np.array([.9, .5, 9.]),StateTraj=np.array([.9, .5, 9.]) ,indexID=7,ParentIndexID=1)

    """Create a Tree (given the tree is tested!!):"""
    # computCostToCome is preferred to be done explicitly; to mack sure we only compute it after establishing an edge with a parent
    tree = ExpansionTree()
    tree.addVertex(v_0)
    v_0.computeCostToCome(tree)
    tree.addVertex(v_1)
    v_1.computeCostToCome(tree)
    tree.addVertex(v_2)
    v_2.computeCostToCome(tree)
    tree.addVertex(v_3)
    v_3.computeCostToCome(tree)
    tree.addVertex(v_4)
    v_4.computeCostToCome(tree)
    tree.addVertex(v_5)
    v_5.computeCostToCome(tree)
    tree.addVertex(v_6)
    v_6.computeCostToCome(tree)
    # tree.addVertex(v_8)
    # tree.addVertex(v_9)
    # tree.addVertex(v_10)

    #Exercise:
    #Fine the Neares vertex to a sample where I'll put the v_sample directly
    CBF_RRT_object = CBF_RRTstrr()
    CBF_RRT_object.TreeT = tree
    v_nearest = CBF_RRT_object.Nearest(xy_sample=np.vstack([.9, .5]))
    #Add the v_sample (that we just got its nearest vertex)
    tree.addVertex(v_sample)
    v_sample.computeCostToCome(tree)
    # Find the near set:
    Nnear_vSet = CBF_RRT_object.Near(Vertex=v_sample,b_raduis=1.1)
    result_bestParent = CBF_RRT_object.ChooseParent(Nnear_vSet=Nnear_vSet,v_nearest= v_nearest,v_new=v_sample)

    #Verify:
    true_bestParent  = v_0
    assert result_bestParent == true_bestParent


#TODO (Test): Design  a randomized test for ChooseParent (will be of a great use afterwards)
def test_ChooseParent_output_random():
    """Test the returned least expensive parent of the input vertex. The vertex and the near vertices are randomly generated. """


    assert True

def test_ChooseParent_output_with_obstacle():
    assert True
'''
------------------------------------------------------------------------------------------------------------------------
Testing Functions for the Rewire Method: 
------------------------------------------------------------------------------------------------------------------------
'''
def test_Rewire_output_determenstic():
    """Test the attempts of rewiring the near vertices to the recently connected v_new to its best parent. In other words,
      test the attempts of trying to make v_new the parent of each of v_near in order to get lower cost."""

    #Setup:
    params = Params('configs.ini')

    v_nearest0  = Vertex(State=np.vstack([0,0,9]),StateTraj=np.vstack([0,0,9]),indexID=0)
    v_new       = Vertex(State=np.vstack([3,1,9]),StateTraj=np.vstack([3,1,9]),indexID=5,ParentIndexID=0)
    v_near1     = Vertex(State=np.vstack([0,3,9]),StateTraj=np.vstack([0,3,9]),indexID=1,ParentIndexID=0)
    v_near2     = Vertex(State=np.vstack([3,0,9]),StateTraj=np.vstack([3,0,9]),indexID=2,ParentIndexID=0)
    v_near3     = Vertex(State=np.vstack([3,-3,9]),StateTraj=np.vstack([3,-3,9]),indexID=3,ParentIndexID=0)
    v_near4     = Vertex(State=np.vstack([5,3,9]),StateTraj=np.vstack([5,3,9]),indexID=4,ParentIndexID=1)
    v_near5 = Vertex(State=np.vstack([6.0, 4.0, 9]), StateTraj=np.vstack([5 + 1., 3 + 1., 9]), indexID=6,
                     ParentIndexID=4)

    tree  = ExpansionTree()
    tree.addVertex(v_nearest0)
    v_nearest0.computeCostToCome(tree) #The Root Vertex
    tree.addVertex(v_new)
    v_new.computeCostToCome(tree)
    tree.addVertex(v_near1)
    v_near1.computeCostToCome(tree)
    tree.addVertex(v_near2)
    v_near2.computeCostToCome(tree)
    tree.addVertex(v_near3)
    v_near3.computeCostToCome(tree)
    tree.addVertex(v_near4)
    v_near4.computeCostToCome(tree) #Now the tree is connected.
    tree.addVertex(v_near5)
    v_near5.computeCostToCome(tree)  # Now the tree is connected.

    CBF_RRTstrr_obj = CBF_RRTstrr()

    CBF_RRTstrr_obj.TreeT = tree
    Nnear_vSet = [v_nearest0,v_near1,v_near2,v_near3,v_near4]

    #Exercise

    #For now there's no need for the Rewire to return anything; it is just required to update the vertices that have been rewired
    CBF_RRTstrr_obj.addChild(vParent=v_nearest0, vChild=v_new)
    CBF_RRTstrr_obj.addChild(vParent=v_nearest0, vChild=v_near1)
    CBF_RRTstrr_obj.addChild(vParent=v_nearest0, vChild=v_near2)
    CBF_RRTstrr_obj.addChild(vParent=v_nearest0, vChild=v_near3)
    CBF_RRTstrr_obj.addChild(vParent=v_near4, vChild=v_near5)

    CBF_RRTstrr_obj.Rewire(Nnear_vSet=Nnear_vSet,v_new=v_new,v_nearest=v_nearest0)
    result_tree = CBF_RRTstrr_obj.TreeT
    result_verticesSet  = result_tree.VerticesSet
    result_ParentIndexIDList = []

    result_verticesSetKeys = result_verticesSet.keys()
    for key in result_verticesSetKeys:
        if key>0:
            vertex =  result_tree.VerticesSet[key]
            result_ParentIndexIDList.append(vertex.ParentIndexID)
    #Verify
    true_ParentIndexIDlist = [0,0,0,5,0,4] #v_near4 is rewired from v_near1 (indexID=1) to v_new (indexID = 5)
    assert_array_almost_equal(true_ParentIndexIDlist,result_ParentIndexIDList)

def test_Rewire_tree_costs_after_rewiring():
    """
    Test the CostToCome of the leaves of a vertex that has been rewired to a better parent.
    :return:
    """
    params = Params('configs.ini')

    v_nearest0 = Vertex(State=np.vstack([0, 0, 9]), StateTraj=np.vstack([0, 0, 9]), indexID=0)
    v_new = Vertex(State=np.vstack([3, 1, 9]), StateTraj=np.vstack([3, 1, 9]), indexID=5, ParentIndexID=0)
    v_near1 = Vertex(State=np.vstack([0, 3, 9]), StateTraj=np.vstack([0, 3, 9]), indexID=1, ParentIndexID=0)
    v_near2 = Vertex(State=np.vstack([3, 0, 9]), StateTraj=np.vstack([3, 0, 9]), indexID=2, ParentIndexID=0)
    v_near3 = Vertex(State=np.vstack([3, -3, 9]), StateTraj=np.vstack([3, -3, 9]), indexID=3, ParentIndexID=0)
    v_near4 = Vertex(State=np.vstack([5, 3, 9]), StateTraj=np.vstack([5, 3, 9]), indexID=4, ParentIndexID=1)
    v_near5 = Vertex(State=np.vstack([5+1, 3+1, 9]), StateTraj=np.vstack([5+1, 3+1, 9]), indexID=5, ParentIndexID=4)

    tree = ExpansionTree()
    tree.addVertex(v_nearest0)
    v_nearest0.computeCostToCome(tree)  # The Root Vertex
    tree.addVertex(v_new)
    v_new.computeCostToCome(tree)
    tree.addVertex(v_near1)
    v_near1.computeCostToCome(tree)
    tree.addVertex(v_near2)
    v_near2.computeCostToCome(tree)
    tree.addVertex(v_near3)
    v_near3.computeCostToCome(tree)
    tree.addVertex(v_near4)
    v_near4.computeCostToCome(tree)  # Now the tree is connected.
    preRewiringCost_v_near4 = v_near4.CostToCome

    tree.addVertex(v_near5)
    v_near5.computeCostToCome(tree)





    CBF_RRTstrr_obj = CBF_RRTstrr()
    CBF_RRTstrr_obj.TreeT = tree
    CBF_RRTstrr_obj.addChild(vParent=v_nearest0,vChild=v_new)
    CBF_RRTstrr_obj.addChild(vParent=v_nearest0, vChild=v_near1)
    CBF_RRTstrr_obj.addChild(vParent=v_nearest0, vChild=v_near2)
    CBF_RRTstrr_obj.addChild(vParent=v_nearest0, vChild=v_near3)
    CBF_RRTstrr_obj.addChild(vParent=v_near4, vChild=v_near1)
    CBF_RRTstrr_obj.addChild(vParent=v_near4, vChild=v_near5)

    Nnear_vSet = [v_nearest0, v_near1, v_near2, v_near3, v_near4, v_near5]

    # Exercise

    # For now there's no need for the Rewire to return anything; it is just required to update the vertices that have been rewired
    CBF_RRTstrr_obj.Rewire(Nnear_vSet=Nnear_vSet, v_new=v_new, v_nearest=v_nearest0)
    result_costToCome_v_near5 = CBF_RRTstrr_obj.TreeT.VerticesSet[5].CostToCome

    #Verify
    true_costToCome_v_near5[0] = CBF_RRTstrr_obj.TreeT.VerticesSet[4].CostToCome+CBF_RRTstrr_obj.Cost_v2v(v_near4,v_near5)

    assert true_costToCome_v_near5 == result_costToCome_v_near5


'''
------------------------------------------------------------------------------------------------------------------------
Testing Functions for the UpdateCostToCome Method: 
------------------------------------------------------------------------------------------------------------------------
'''
def test_UpdteCostToCome_output():
    """
    Check the CostToCome of the leaves of a root vertex that its CostToCome has changed.
     """

    #Setup
    params = Params('configs.ini')
    TreeT = ExpansionTree()
    CBF_RRT_object = CBF_RRTstrr(x_g=np.vstack([30., 30.]), eps_g=5,TreeT=TreeT)
    v_0 = Vertex(State=np.array([0., 0., 9.]), indexID=0, TreeT=TreeT)
    CBF_RRT_object.addVertex(v_0)
    v_0.CostToCome = .1

    v_1 = Vertex(State=np.array([1., 1., 9.]), indexID=1,ParentIndexID=0, TreeT=TreeT)
    CBF_RRT_object.addVertex(v_1)
    v_1.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    v_2 = Vertex(State=np.array([1., 0., 9.]), indexID=2,ParentIndexID=0, TreeT=TreeT)
    CBF_RRT_object.addVertex(v_2)
    v_2.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    v_3 = Vertex(State=np.array([1.,- 1., 9.]), indexID=3,ParentIndexID=0, TreeT=TreeT)
    CBF_RRT_object.addVertex(v_3)
    v_3.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    v_4 = Vertex(State=np.array([1., 2., 9.]), indexID=4,ParentIndexID=1, TreeT=TreeT)
    CBF_RRT_object.addVertex(v_4)
    v_4.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    v_5 = Vertex(State=np.array([2., 1., 9.]), indexID=5,ParentIndexID=1, TreeT=TreeT)
    CBF_RRT_object.addVertex(v_5)
    v_5.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    v_6 = Vertex(State=np.array([2., 0., 9.]), indexID=6,ParentIndexID=2,TreeT=TreeT)
    CBF_RRT_object.addVertex(v_6)
    v_6.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    v_7 = Vertex(State=np.array([2., -1., 9.]), indexID=7, ParentIndexID=3,TreeT=TreeT)
    CBF_RRT_object.addVertex(v_7)
    v_7.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    v_8 = Vertex(State=np.array([3., 2., 9.]), indexID=8, ParentIndexID=5,TreeT=TreeT)
    CBF_RRT_object.addVertex(v_8)
    v_8.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    v_9 = Vertex(State=np.array([3., 0., 9.]), indexID=9,ParentIndexID=7,TreeT=TreeT)
    CBF_RRT_object.addVertex(v_9)
    v_9.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    v_10 = Vertex(State=np.array([3., -1., 9.]), indexID=10,ParentIndexID=7 ,TreeT=TreeT)
    CBF_RRT_object.addVertex(v_10)
    v_10.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    v_11 = Vertex(State=np.array([3., -2., 9.]), indexID=11, ParentIndexID=7,TreeT=TreeT)
    CBF_RRT_object.addVertex(v_11)
    v_11.computeCostToCome(TreeT=CBF_RRT_object.TreeT)

    CBF_RRT_object.addChild(v_0,v_1)
    CBF_RRT_object.addChild(v_0, v_2)
    CBF_RRT_object.addChild(v_0, v_3)

    CBF_RRT_object.addChild(v_1, v_4)
    CBF_RRT_object.addChild(v_1, v_5)
    CBF_RRT_object.addChild(v_2, v_6)
    CBF_RRT_object.addChild(v_3, v_7)

    CBF_RRT_object.addChild(v_5, v_8)
    CBF_RRT_object.addChild(v_7, v_9)
    CBF_RRT_object.addChild(v_7, v_10)
    CBF_RRT_object.addChild(v_7, v_11)


    #Exercise
    result_costToComeList = []
    v_0.CostToCome = 1
    CBF_RRT_object.UpdateCostToCome(v_0,1)
    for vertex in CBF_RRT_object.TreeT.VerticesSet.values():
        result_costToComeList.append(vertex.CostToCome)

    #Verify
    true_CostToComeListBeforeUpdate = [0,1.4142,1,1.4142, 2.4142,2.4142,2,2.4142,3.828,3.828,3.4142,3.828]
    true_CostToComeListAfterUpdate = list(np.array(true_CostToComeListBeforeUpdate)+1)

    assert_array_almost_equal(result_costToComeList,true_CostToComeListAfterUpdate,3)


########################################################################################################################
#################################### ExpansionTree Tests: ##############################################################
########################################################################################################################

def test_addVertex_outputDim():
    """Test the dimension of the VertexSet dict as well as the xyCoorVertices after adding a vertex to the Tree"""

    #Setup
    params = Params('configs.ini')
    tree = ExpansionTree()

    #Exercise
    for i in range(10):
        v = Vertex(State=np.vstack([1,i,1]),indexID=i+2)
        tree.addVertex(v)
    result_vertexSet_size = len(tree.VerticesSet)
    #Verify
    true_vertexSet_size   = 10
    assert true_vertexSet_size == result_vertexSet_size

def test_addVertex_output():
    assert True
def test_updateVertex_outputDim():
     assert True
def test_updateVertex_output():
    assert True






########################################################################################################################
#################################### Adaptive Sampling Tests: ###########################################################
########################################################################################################################
'''
------------------------------------------------------------------------------------------------------------------------
Testing Functions for the Kernel-estimators 
------------------------------------------------------------------------------------------------------------------------
'''
def test_fitting_synthetic_GMM_datatPoints_to_the_KDestimate():
    """
    Test the discrepancy between ground-truth GMM synthetic data points with the Kernel density estimate
    :return:
    """

    #Setup

    #generating synthetic datapoints:
    #The covariance matrix:
    x = np.zeros(100)
    y = np.zeros(100)
    for iSample in range(100):
        theta = np.pi/4
        u1_T = [np.cos(theta),np.sin(theta)]
        u2_T = [np.sin(theta),-np.cos(theta)]
        U_T = np.array([u1_T,u2_T])
        U = U_T.transpose()
        sigma1 = 30
        sigma2 = 10
        Lambda = np.array([[sigma1,0],[0,sigma2]])
        cov = U.dot(Lambda.dot(U.transpose()))
        mean = np.array([6 , 6])
        x[iSample],y[iSample]=np.random.multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)

    plt.scatter(x,y)
    plt.show()
    a = 2

    assert True







'''
------------------------------------------------------------------------------------------------------------------------
Testing Functions for the Rewire Method: 
------------------------------------------------------------------------------------------------------------------------
'''

#Other Functions
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


