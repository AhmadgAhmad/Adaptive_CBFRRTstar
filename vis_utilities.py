import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import pickle
import numpy as np

class plt_tree_uts:
    def __init__(self) -> None:
        pass
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
            goalRegion = plt.Circle((xy_goal[0], xy_goal[1]), eps_g, linestyle='--', color='g', fill=False)
            ax.add_patch(goalRegion)
            plt.plot(xy_start[0], xy_start[1], "xb", markersize=15, label="Initial State", hold='on')
            plt.plot(xy_goal[0], xy_goal[1], "^r", markersize=15, label="Goal State", hold='on')

            plt.xlabel("$x_1$", fontsize=15)
            plt.ylabel("$x_2$", fontsize=15)
            plt.xticks(size=10)
            plt.yticks(size=10)
            # plt.axis(axislim)
            plt.grid(True)
            plt.legend(loc='best', borderaxespad=0., prop={'size': 10})
            plt.rcParams.update({'font.size': 10})

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


    def plot_pathToVertex(self, vertex, EnPlotting=False, colorPath=[1., 0., 1.]):
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
                plt.plot(vertexTrajectory[0, :], vertexTrajectory[1, :], markersize=.1, color=colorPath, mfc=[1., 0., 1.],
                        mec=[0.1, 0.1, 0.1], hold='on')
                self.TreePlot = plt
            else:
                pass


    def plot_tree(self,v_nearest2point = None,plot_pathFalg = False):
            VerticesSet = self.TreeT.VerticesSet
            VerticesSetValues = VerticesSet.values()

            # Plot all the paths
            for vertex in VerticesSetValues:
                if False:#vertex.vgFlag:
                    continue
                else:
                    self.plot_vertex(vertex, EnPlotting=True)
                    if vertex.ParentIndexID is not None:
                        self.plot_pathToVertex(vertex, EnPlotting=True)
                        w = vertex.StateTraj[0,0]-self.TreeT.VerticesSet[vertex.ParentIndexID].State[0]
                        if w>.2:
                            a = 2

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



# Plotting the distribution stuff: 
            #Save the grid points with the corresponding probs, the cost, and the tree to plot them afterwards:
            saveData(self.goal_costToCome_list, 'adapCBF_RRTstr_Cost', suffix=self.suffix, CBF_RRT_strr_obj=self,
                     adapDist_iter=self.adapIter-1, enFlag=False)

            saveData([self.TreeT,self.vg_minCostToCome_list], 'adapCBF_RRTstr_Tree', suffix=self.suffix, CBF_RRT_strr_obj=self,
                     adapDist_iter=self.adapIter - 1, enFlag=False)
            saveData([Xxgrid, Xygrid, grid_probs.reshape(Xxgrid.shape),elite_samples_arr], 'adapCBF_RRTstr_KDEgridProbs',
                     suffix=self.suffix, CBF_RRT_strr_obj=self,
                     adapDist_iter=self.adapIter - 1, enFlag=False)

            #Plot the distribution
            if self.params.plot_pdf_kde:
                self.initialize_graphPlot()
                CS = plt.contour(Xxgrid, Xygrid, grid_probs.reshape(Xxgrid.shape))  # , norm=LogNorm(vmin=4.18, vmax=267.1))
                # plt.colorbar(CS, shrink=0.8, extend='both')
                plt.scatter(elite_samples_arr[:, 0], elite_samples_arr[:, 1])
                plt.show()