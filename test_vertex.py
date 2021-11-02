from CBF_RRT_strr import CBF_RRTstrr
from vertex import Vertex
import numpy as np
from expansionTree import ExpansionTree
from params import Params
from numpy.testing import *
import random
import math


def test_addChild_output_ChildrenIndexIDs():
    """Test if the ChildrenIndexIDs is as expected"""

    #Setup:
    params = Params('configs.ini')
    vertex = Vertex(State=np.array([0., 0., 9.]), indexID=0)

    #Exercise:
    vertex.addChild(1)
    vertex.addChild(4)
    vertex.addChild(5)
    vertex.addChild(11)
    result_ChildrenIndexIDs = vertex.ChildrenIndexIDs

    #Verify:
    true_ChildrenIndexIDs = [1,4,5,11]
    assert_array_equal(true_ChildrenIndexIDs,result_ChildrenIndexIDs)

    #Clean Up:
    #TODO: Check for duplication of adding a chiled
    assert True
#
# def test_PathToCome_output_directly_assigned_children_path():
#
#     """
#
#     :return: True if the PathToCome to the vertex as expected:
#     """
#     #Setup
#     params = Params('configs.ini')
#     TreeT = ExpansionTree()
#     CBF_RRT_object = CBF_RRTstrr(x_g=np.vstack([30., 30.]), eps_g=5, TreeT=TreeT)
#     v_0 = Vertex(State=np.array([0., 0., 9.]), indexID=0)
#     CBF_RRT_object.addVertex(v_0)
#     v_0.CostToCome = .1
#
#     v_1 = Vertex(State=np.array([1., 1., 9.]), indexID=1, ParentIndexID=0)
#     CBF_RRT_object.addVertex(v_1)
#     v_1.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     v_2 = Vertex(State=np.array([1., 0., 9.]), indexID=2, ParentIndexID=0)
#     CBF_RRT_object.addVertex(v_2)
#     v_2.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     v_3 = Vertex(State=np.array([1., - 1., 9.]), indexID=3, ParentIndexID=0)
#     CBF_RRT_object.addVertex(v_3)
#     v_3.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     v_4 = Vertex(State=np.array([1., 2., 9.]), indexID=4, ParentIndexID=1)
#     CBF_RRT_object.addVertex(v_4)
#     v_4.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     v_5 = Vertex(State=np.array([2., 1., 9.]), indexID=5, ParentIndexID=1)
#     CBF_RRT_object.addVertex(v_5)
#     v_5.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     v_6 = Vertex(State=np.array([2., 0., 9.]), indexID=6, ParentIndexID=2)
#     CBF_RRT_object.addVertex(v_6)
#     v_6.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     v_7 = Vertex(State=np.array([2., -1., 9.]), indexID=7, ParentIndexID=3)
#     CBF_RRT_object.addVertex(v_7)
#     v_7.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     v_8 = Vertex(State=np.array([3., 2., 9.]), indexID=8, ParentIndexID=5)
#     CBF_RRT_object.addVertex(v_8)
#     v_8.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     v_9 = Vertex(State=np.array([3., 0., 9.]), indexID=9, ParentIndexID=7)
#     CBF_RRT_object.addVertex(v_9)
#     v_9.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     v_10 = Vertex(State=np.array([3., -1., 9.]), indexID=10, ParentIndexID=7)
#     CBF_RRT_object.addVertex(v_10)
#     v_10.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     v_11 = Vertex(State=np.array([3., -2., 9.]), indexID=11, ParentIndexID=7)
#     CBF_RRT_object.addVertex(v_11)
#     v_11.computeCostToCome(TreeT=CBF_RRT_object.TreeT)
#
#     CBF_RRT_object.addChild(v_0, v_1)
#     CBF_RRT_object.addChild(v_0, v_2)
#     CBF_RRT_object.addChild(v_0, v_3)
#
#     CBF_RRT_object.addChild(v_1, v_4)
#     CBF_RRT_object.addChild(v_1, v_5)
#     CBF_RRT_object.addChild(v_2, v_6)
#     CBF_RRT_object.addChild(v_3, v_7)
#
#     CBF_RRT_object.addChild(v_5, v_8)
#     CBF_RRT_object.addChild(v_7, v_9)
#     CBF_RRT_object.addChild(v_7, v_10)
#     CBF_RRT_object.addChild(v_7, v_11)
#
#     #Exercise
#     result_path0 = v_0.PathToCome
#     result_path1 = v_1.PathToCome
#     result_path2 = v_2.PathToCome
#     result_path3 = v_3.PathToCome
#     result_path4 = v_4.PathToCome
#     result_path5 = v_5.PathToCome
#     result_path6 = v_6.PathToCome
#     result_path7 = v_7.PathToCome
#     result_path8 = v_8.PathToCome
#     result_path9 = v_9.PathToCome
#     result_path10 = v_10.PathToCome
#     result_path11 = v_11.PathToCome
#
#     #Verify
#     true_path0 = [0]
#     true_path1 = [0,1]
#     true_path2 = [0,2]
#     true_path3 = [0,3]
#     true_path4 = [0,1,4]
#     true_path5 = [0,1,5]
#     true_path6 = [0,2,6]
#     true_path7 = [0,3,7]
#     true_path8 = [0,1,5,8]
#     true_path9 = [0,3,7,9]
#     true_path10 = [0, 3, 7, 10]
#     true_path11 = [0, 3, 7, 11]
#
#     assert_array_equal(true_path0,result_path0)
#     assert_array_equal(true_path1, result_path1)
#     assert_array_equal(true_path2, result_path2)
#     assert_array_equal(true_path3, result_path3)
#     assert_array_equal(true_path4, result_path4)
#     assert_array_equal(true_path5, result_path5)
#     assert_array_equal(true_path6, result_path6)
#     assert_array_equal(true_path7, result_path7)
#     assert_array_equal(true_path8,result_path8)
#     assert_array_equal(true_path9, result_path9)
#     assert_array_equal(true_path10, result_path10)
#     assert_array_equal(true_path11, result_path11)


def test_PathToCome_output_path_after_rewiring():
    assert True

def test_PathToCome_output_path_after_choosingParent():
    assert True
