from CBF_RRT_strr import CBF_RRTstrr
from vertex import Vertex
import numpy as np
from expansionTree import ExpansionTree
from params import Params
from numpy.testing import *






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
    NN_distance = 100000
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


#TODO: Do the random test.
def test_Near_return_return_near_vertecSet_Random():
    """Test if Nearest will return the nearest vertex of randomly sampled vertices as well as random testing point."""
    #Setup
    Params('configs.ini')
    tree = ExpansionTree()
    CBF_RRTstrr_object = CBF_RRTstrr()
    #Random sample:
    xy_sample = np.vstack([np.random.uniform(high=10),np.random.uniform(high=10)])
    NN_distance = 100000
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


    #Exercise
    result_nnVertex = CBF_RRTstrr_object.Nearest(xy_sample,TreeT=tree)

    #Verify
    true_nnVertex = tree.VerticesSet[true_nn_indexID]
    assert result_nnVertex==true_nnVertex


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
    v_1tTraj  = .1 * v_1Traj[0,:]
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
    true_bestParent  = v_1
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
    v_nearest0  = Vertex(State=np.vstack([0,0,9]),StateTraj=np.vstack([0,0,9]),indexID=0)
    v_new       = Vertex(State=np.vstack([3,1,9]),StateTraj=np.vstack([3,1,9]),indexID=5,ParentIndexID=0)
    v_near1     = Vertex(State=np.vstack([0,3,9]),StateTraj=np.vstack([0,3,9]),indexID=1,ParentIndexID=0)
    v_near2     = Vertex(State=np.vstack([3,0,9]),StateTraj=np.vstack([3,0,9]),indexID=2,ParentIndexID=0)
    v_near3     = Vertex(State=np.vstack([3,-3,9]),StateTraj=np.vstack([3,-3,9]),indexID=3,ParentIndexID=0)
    v_near4     = Vertex(State=np.vstack([5,3,9]),StateTraj=np.vstack([5,3,9]),indexID=4,ParentIndexID=1)

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
    CBF_RRTstrr_obj = CBF_RRTstrr()
    CBF_RRTstrr_obj.TreeT = tree
    Nnear_vSet = [v_nearest0,v_near1,v_near2,v_near3,v_near4]

    #Exercise

    #For now there's no need for the Rewire to return anything; it is just required to update the vertices that have been rewired
    CBF_RRTstrr_obj.Rewire(Nnear_vSet=Nnear_vSet,v_new=v_new,v_nearest=v_nearest0)
    result_tree = CBF_RRTstrr_obj.TreeT
    result_verticesSet  = result_tree.VerticesSet
    result_ParentIndexIDList = []
    for vertex in result_verticesSet:
        result_ParentIndexIDList.append(vertex.ParentIndexID)
    #Verify
    true_ParentIndexIDlist = [0,0,0,5] #v_near4 is rewired from v_near1 (indexID=1) to v_new (indexID = 5)
    assert_array_almost_equal(true_ParentIndexIDlist,result_ParentIndexIDList)


#TODO (Test): Design  a randomized test for Rewire procedure (will be of a great use afterwards)
def test_Rewire_output_random():
    assert True

def test_Rewire_output_with_obstacle():
    assert True

########################################################################################################################
#################################### Adaptive Sampling Tests: ###########################################################
########################################################################################################################

'''
------------------------------------------------------------------------------------------------------------------------
Testing Functions for the Rewire Method: 
------------------------------------------------------------------------------------------------------------------------
'''




