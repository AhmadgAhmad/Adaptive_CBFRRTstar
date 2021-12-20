import sys
print(sys.version)
import time
import timeit
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




class Simulation(object):

    def __init__(self):
        self.params = Params()
        self.agents = list()
        self.obsts = list()
        self.cur_timestep = 0
        self.max_timesteps = self.params.max_timesteps
        self.time_vec = [0]

        # CBF Only Fields
        self.u_refs = list()

        # CLF Fields
        self.epsilon = self.params.epsilon

        # The recorded step size dt
        self.r_dt = self.params.step_size
        # Plotting setup
        if self.params.plot_sim:
            self.sim_fig, self.sim_axes = plt.subplots()
            self.xlim = None
            self.ylim = None

    def add_agent(self, agent):
        # Priority values work best in range [0.25, 1.75]
        agent.id = len(self.agents)
        self.agents.append(agent)

    def add_obstacle(self, obst):
        obst.id = len(self.obsts)
        self.obsts.append(obst)

    #Plotting the simulation scenarios:
    def plot_scenario(self):
        self.sim_axes.cla()
        [ag.plot(self.sim_axes) for ag in self.agents]
        [ob.plot(self.sim_axes) for ob in self.obsts]

        if self.xlim is None:
            self.xlim = plt.xlim()
            self.ylim = plt.ylim()
        else:
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
        plt.pause(.001)

    def add_cbf_pair(self, m, agt, obst):
        m.update()

        if agt.dyn_enum is Dyn.UNICYCLE:
            #Add the CBF constraints directly without going through methods in other classes:
            if self.params.hocbf_ufn_en:
                hocbf.Update_m()
            else:
                # q_dot = agt.get_x_dot()[0:2] - obst.get_x_dot()[0:2]
                # q_dot = agt.get_q_dot()[0:2] #TODO [speedQ]
                # p1 = self.params.alpha_linear_p1
                #alpha11_h = p1 * obst.h.eval(agt)
                if isinstance(obst,Sphere):
                    alpha1_h = p1 *((agt.state[0:2] - obst.state[0:2]).T.dot(obst.M).dot((agt.state[0:2] - obst.state[0:2]))[0][0] - (
                            obst.sign * agt.radius + obst.radius) ** 2) * obst.sign
                    grad_h = (2 * obst.M.dot((agt.state[0:2] - obst.state[0:2]))) * obst.sign
                    psi1 = grad_h.T.dot(q_dot[0:2]) + alpha1_h
                # 6-3-2021:
                    p1 = 1
                    p2 = 2
                    h = (agt.state[0] - obst.state[0])**2+(agt.state[1] - obst.state[1])**2 - (obst.radius+.7)**2
                    Lfh = 2*agt.u[0]*(agt.state[0]-obst.state[0])*np.cos(agt.dyn.cur_theta)+\
                        2*agt.u[0]*(agt.state[1]-obst.state[1])*np.sin(agt.dyn.cur_theta)
                    psi1 = 2*(agt.u[0]**2)*agt.state[0]*(np.cos(agt.dyn.cur_theta)**2)+ \
                        2 * (agt.u[0] ** 2) *agt.state[1]*(np.sin(agt.dyn.cur_theta)**2)+\
                        (2*agt.u[0]*(agt.state[1]-obst.state[1])*np.cos(agt.dyn.cur_theta)-
                        2*agt.u[0]*(agt.state[0]-obst.state[0])*np.sin(agt.dyn.cur_theta))*agt.u[1]+\
                        (p1)*h+(p2)*Lfh


                else: #For ellipsoid obstacle:
                    a_aug = obst.a + ((agt.radius  * obst.sign)*1)
                    b_aug = obst.b + ((agt.radius  * obst.sign)*1)
                    theta1 = np.radians(obst.angle)

                    c = np.cos(theta1)
                    s = np.sin(theta1)
                    s = c

                    aa = (c / a_aug) ** 2 + (s / b_aug) ** 2
                    bb = s * c * ((1 / b_aug )** 2 - (1 / a_aug) ** 2)
                    cc = (s / a_aug) ** 2 + (c / b_aug) ** 2
                    M_ellip = np.array([[aa, bb], [bb, cc]])
                    #alpha11_h = p1 * obst.h.eval(agt)
                    # h = ((agt.state[0:2] - obst.state[0:2]).transpose().dot(M_ellip).dot((agt.state[0:2] - obst.state[0:2]))[0][0] - 1) * obst.sign
                    # grad_h = (2 * M_ellip.dot((agt.state[0:2] - obst.state[0:2]))) * obst.sign
                    # psi1 = grad_h.T.dot(q_dot[0:2]) + alpha1_h
                    # 6-3-2021 (Doing the computation this way is faster when executing the code):
                    M1 = aa
                    Mb = 2*bb
                    M2 = cc
                    p1 = -1
                    p2 = .5

                    #good
                    p1 = 10
                    p2 = .5
                    
                    p1 = 109
                    p2 = 30

                    # p1 = 55.0
                    # p2 = 30.0

                    h = (M1*(agt.state[0] - obst.state[0])**2+Mb*(agt.state[0] - obst.state[0])*(agt.state[1] - obst.state[1])+\
                        (M2*(agt.state[1] - obst.state[1])**2)-1)
                    # h = (agt.state[0:2] - obst.state[0:2]).transpose().dot(M_ellip).dot((agt.state[0:2] - obst.state[0:2]))[0][0] - 1) * obst.sign
                    Lfh = ((agt.state[0] - obst.state[0]) * (2*M1*agt.u[0]*np.cos(agt.state[2]) + Mb*agt.u[0]* np.sin(agt.state[2])))+ \
                        ((agt.state[1] - obst.state[1]) * (2 * M2 * agt.u[0] * np.sin(agt.state[2]) + Mb * agt.u[0] * np.cos(agt.state[2])))
                    # psi1 = 2*M1*(agt.u[0]*np.cos(agt.dyn.cur_theta))**2 + 2*M2*(agt.u[0]*np.sin(agt.dyn.cur_theta))**2+2*Mb*(agt.u[0]**2)*np.cos(agt.dyn.cur_theta)*np.sin(agt.dyn.cur_theta) +agt.u[0]*(-2*M1*np.sin(agt.dyn.cur_theta)*(agt.state[0] - obst.state[0])-Mb*(agt.state[1] - obst.state[1])*np.sin(agt.dyn.cur_theta)+Mb*(agt.state[0] - obst.state[0])*np.cos(agt.dyn.cur_theta)+2*M2*np.cos(agt.dyn.cur_theta) *(agt.state[1] - obst.state[1])) * agt.u[1]+\
                    #                  +(p1)*Lfh + (p2)*h

                    sr  = np.sin(agt.state[2])
                    cr  = np.cos(agt.state[2])
                    v = agt.u[0]
                    w = agt.u[1]
                    psi1 = (agt.state[0] - obst.state[0]) * (-2*M1*v*sr+Mb*v*cr) * w +\
                        (2*M1*v*cr+Mb*v*sr)*v*cr + \
                        (agt.state[1] - obst.state[1]) * (2 * M2 * v * cr - Mb * v * sr) * w + \
                        (2 * M2 * v * sr + Mb * v * cr) * v * sr + \
                        (p1+p2) * Lfh + (p2*p1) * h
                        #Compute the gradient:
                m.addConstr((psi1[0] >= 0.0), name="CBF_{}".format(agt.id))
            # t1_adcbf = timeit.default_timer()
            # T_adcbf = t1_adcbf - t0_adcbf
        
        elif agt.dyn_enum is Dyn.SINGLE_INT:
            if self.params.decentralized:
                obst_x_dot = obst.get_x_dot((0,0))
                k_cbf = obst.k_cbf
                k_cbf = 0.5
            else:
                obst_x_dot = obst.get_x_dot()
                k_cbf = 2
            p_cbf = 1.0

            h_val = obst.h.eval(agt)
            lg_h = obst.h.grad(agt).T.dot(agt.get_q_dot() - obst_x_dot)[0][0]
            constr = m.addConstr((lg_h)>=-k_cbf*h_val**p_cbf, name="CBF_{}".format(agt.id))
            attr = GRB.Attr.RHS
            m.update()


       # hocbf.get_psi2()
        if self.params.plot_cbf:
            pass
            # h_val = hocbf.get_hocbf_val()
            #h_val = obst.h.eval(agt)
            # self.cbf_data.add_cbf_val(h_val, agt, obst)
        #TODO: update hocbf to get the constraint value needed.
        #It is important if you want to probe when the constraint is active.
        if self.params.plot_constrs:
            pass
            # self.cbf_data.add_constr_val(constr.getAttr(attr),agt,obst)
    def agt_ellipD(self,agt,eobst):
        """
        Checks if an elliptic obstacle lies within the sensing area of the agent
        :param eobst: elliptic obstacle
        :return: True if eobst lies within the sensing circular footprint of the agent
        """

        #The ellipsoid parameters and state:
        a = eobst.a
        b = eobst.b
        thetaE = eobst.angle
        c = math.cos(thetaE/57.3)
        s = math.sin(thetaE/57.3)
        xo = eobst.state[0]
        yo = eobst.state[1]

        #Rotate the center of the state of the agent:
        xa = agt.state[0]
        ya = agt.state[1]
        ra_sensor = agt.footPrint_radi+.5

        xa_rot = c*xa + s*ya
        ya_rot = -s*xa + c*ya
        #Compute the angle betwee the center of the ellips and the center of the agent footprint circle:
        dx = xa_rot - xo
        dy = ya_rot - yo
        angle = math.atan2(-dy, dx)
        x, y = eobst.pointFromAngle(angle)
        distance = math.hypot(x - xa_rot, y - ya_rot)
        return distance <= ra_sensor

    def agt_circD(self,agt,cobst):
        """
        Return true if there's a collision between the robot footprint and circular obstacle.
        :param agt:
        :param cobst:
        :return:
        """
        xo = cobst.state[0]
        yo = cobst.state[1]
        if isinstance(cobst, Sphere):
            ro = cobst.radius
        else:
            ro = cobst.a

        xa = agt.state[0]
        ya = agt.state[1]
        
        ra_sensor = agt.footPrint_radi

        d = math.sqrt((xo-xa)**2+(yo-ya)**2)
        return d <= (ra_sensor+ro)

    def setup_plots(self):
        clf_used = False
        for agt in self.agents:
            if agt.goal is not None:
                clf_used = True
                break
        if not clf_used:
            self.params.plot_clf=False
            self.params.plot_delta=False

        if len(self.agents)==1 and len(self.obsts)==0:
            self.params.plot_cbf = False
            self.params.plot_constrs =False

        if self.params.plot_clf or self.params.plot_delta:
            self.clf_data = Clf_data(self.agents)
        if self.params.plot_cbf or self.params.plot_constrs:
            self.cbf_data = Cbf_data(self.agents,self.obsts)

        # Set agent colors
        cm = ColorManager()
        for i in range(len(self.agents)):
            self.agents[i].color = cm.get_colors(i)
            

    def show_plots(self, save=False):
        if self.params.plot_clf or self.params.plot_delta:
            self.clf_data.plot(self.time_vec,save=save)
        if self.params.plot_cbf or self.params.plot_constrs:
            self.cbf_data.plot(self.time_vec,save=save)
        
        plt.rcParams.update({'font.size': 18})
        plt.show()

    def goalReached(self,retrackAbortFlag):
        goal_reached = False
        if self.cur_timestep == self.max_timesteps:
            goal_reached = True

        if not goal_reached:
            done=[]
            [done.append(agt.done) for agt in self.agents]
            if all(done):
                goal_reached = True

        if goal_reached:
            pass
            #self.show_plots(save=True)

        if self.params.plot_sim:
            with PdfPages('z_scenario.pdf') as pdf:
                    pdf.savefig(self.sim_fig)

        if retrackAbortFlag:
            goal_reached = True

        return goal_reached

    def initiate(self, steps=None):
        # Setup plots
        self.setup_plots()

        if steps is not None:
            x_sol = np.zeros((len(self.agents), 2,steps))
            u_sol = np.zeros((len(self.agents), 2,steps))

        m = Model("CLF_CBF_QP")

        #Stop optimizer from publishing results to console
        m.Params.LogToConsole = 0

        goal_reached = False

        while not goal_reached:
            # Remove constraints and variables from previous loop
            m.remove(m.getConstrs())
            m.remove(m.getQConstrs())
            m.remove(m.getVars())

            for idx in range(len(self.agents)):
                # Get agent for this loop
                agt = self.agents[idx]
                #get u_ref from the goal in here!! or
                # Add the Gurobi control variables
                agt.add_control(m,idx)
                # If goal is specified, add CLF constraints
                if agt.goal is not None:

                    # Control Lyapunov Function (Creates control variables for Model m)
                    v_val = agt.add_clf(m)
                    if self.params.plot_clf:
                        self.clf_data.add_clf_val(v_val, agt.id)

                # Otherwise, apply u_ref for the agent
                else:
                    u_ref = agt.u_ref
                    #TODO: Automate how u_ref is chosen based on the choice of the enabled controls!
                    if u_ref is not None:
                        # Add CBF objective function for this agent
                        #u_ref_t = make_column(u_ref[:,self.cur_timestep])
                        if agt.dyn_enum is Dyn.UNICYCLE: #TODO (RRTbug)
                            if len(u_ref[1,:])==1:
                                curTheta = agt.dyn.cur_theta
                                u_ref_t = make_column([u_ref[0,0], u_ref[1,0] - curTheta])
                            else:
                                # Define u_ref as the negative gradient of a potential function  
                                curTheta = agt.dyn.cur_theta
                                # u_ref_t = make_column([u_ref[0, self.cur_timestep],u_ref[1, self.cur_timestep] -curTheta]) 
                                u_ref_t = np.array([u_ref[0, self.cur_timestep],u_ref[1, self.cur_timestep] -curTheta])
                        elif agt.dyn_enum is Dyn.UNICYCLE_EXT:
                            curTheta = agt.dyn.cur_state[2]
                            u_ref_t = make_column([u_ref[0, self.cur_timestep]-curTheta, u_ref[1, self.cur_timestep]])
                        elif agt.dyn_enum is Dyn.SINGLE_INT: #TODO (SnglInt Exte)
                            curTheta = agt.dyn.cur_state[1]
                            # u_ref_t = make_column([u_ref[0, self.cur_timestep] - curTheta, u_ref[1, self.cur_timestep]])
                            u_ref_t = make_column(np.array([u_ref[0, self.cur_timestep],u_ref[1, self.cur_timestep]]))


                        cost_func = (agt.u - u_ref_t).T.dot((agt.u - u_ref_t))[0][0] #
                        m.setObjective(m.getObjective() + cost_func, GRB.MINIMIZE)
                        m.update()

            # Add Pairwise CBF Constraints
            for i in range(len(self.agents)):
                agt = self.agents[i]

                if self.params.decentralized:
                    agt2 = range(len(self.agents))
                    agt2.remove(i)
                else:
                    agt2 = range(i+1, len(self.agents))
                
                # CBF agent/agent
                for j in agt2:
                    self.add_cbf_pair(m, agt, self.agents[j])

                t0_obs_cbfs = timeit.default_timer()
                # CBF agent/obstacle
                # Checking the obstacles in robot's sensor footprint is done here

                for k in range(len(self.obsts)):
                    if isinstance(self.obsts[k],Sphere):
                        flag = self.agt_circD(agt,self.obsts[k])
                    else:
                        flag = self.agt_circD(agt, self.obsts[k])
                    if flag:#flag: # If there's an obstacle in the sensor footprint 
                        self.add_cbf_pair(m, agt, self.obsts[k])
                t1_obs_cbfs  = timeit.default_timer()
                T_obs_cbfs = t1_obs_cbfs-t0_obs_cbfs
                a = 1

            # Solve the optimization problem
            #FeasFlagDummy = checkFeasibility(self, m)
            
            m.optimize()
            
            
            #Plot the CBF constraint here: 
            if self.params.plot_delta:
                [self.clf_data.add_delta_val(m.getVarByName("delta{}".format(agt.id)).x, agt.id) for agt in self.agents]


            '''This loop will keep deviding the integration step size if infeasibility of the QP is encountered.
            # This will happen if the step size is long enough to steer the next step inside an obstacle.
            # TODO: refer to the ZOH $\tau$-CBF paper. ANALYZE it first with the Unicycle dynamics, the case study there is for double integrator dynamics.
            TODO: refer to HOCLBF with STL Wei, et al.'''
            FeasFlag = False  # Feasibility flag, in order to perform backtracking if encountered infeasibility due to step size
            dt = self.r_dt
            retrackAbortFlag = False
            #------ The checking approach--------------------------------------------------------------------
            # dummyAgent = self.agents[0]
            # dummyAgentU = dummyAgent.dyn.get_u()
            
            if m.Status == 3: #Infeasible
                #TODO (find a better way to abort)
                # retrackAbortFlag = True
                # if dt < .01:
                #     retrackAbortFlag = True
                # else:
                #     retrackAbortFlag = False
                dt = dt / 2
                #Update the recorded dt
                self.r_dt = dt
                #Retrack to the previous state (extract for the incremental trajectory):
                #if dt == 0.05: #WARNING: This is wrong with more then one agent
                    #ag.dyn.trajectory = ag.dyn.trajectory[:, :-1]
                    #ag.dyn.time = ag.dyn.time[:-1]
                for ag in self.agents:
                    if ag.dyn_enum is Dyn.UNICYCLE_EXT:
                        # Truncate the trajectory that has an infeasible final state (man look out for the complexity!!):
                        ag.dyn.trajectory = ag.dyn.trajectory[:, :-1]
                        prvFeasible_state = ag.dyn.trajectory[:, -1]
                        prvFeasible_state = np.vstack(prvFeasible_state)
                        # Truncate the time trajectory:
                        ag.dyn.time = ag.dyn.time[:-1]
                        prvFeasible_state_time = ag.dyn.time[-1]
                        prvFeasible_state_time = np.array(prvFeasible_state_time)
                        #Set the current state as the previous one and the current time as the previous one:
                        ag.dyn.cur_state = prvFeasible_state
                        ag.dyn.cur_time = prvFeasible_state_time
                        #Move one step and we'll see if the problem is still infeasible:
                        ag.step(u=np.vstack(ag.dyn.get_u()), plot=False, time_step=dt)
                    elif ag.dyn_enum is Dyn.UNICYCLE:
                        #---- Truncate the trajectory that has an infeasible final state (man look out for the complexity!!):

                        if len(ag.dyn.trajectory) <=1: 
                            ag.dyn.trajectory = ag.dyn.trajectory
                            prvFeasible_state = ag.dyn.trajectory[0]
                            prvFeasible_state = np.array(prvFeasible_state)
                            # ---- Truncate the time trajectory:
                            ag.dyn.time = ag.dyn.time
                            prvFeasible_state_time = ag.dyn.time
                            prvFeasible_state_time = np.array(prvFeasible_state_time)

                        else:
                            ag.dyn.trajectory = ag.dyn.trajectory[:-1]
                            prvFeasible_state = ag.dyn.trajectory[-1]  #TODO E[speedQ] Reterack to 2 previous steps since the previous step is already very close. 
                            #---- Truncate the time trajectory:
                            ag.dyn.time = ag.dyn.time[:-1]
                            prvFeasible_state_time = ag.dyn.time[-1]
                            prvFeasible_state_time = np.array(prvFeasible_state_time)

                        #-----Set the current state, theta, and time as the previous one:
                        ag.dyn.cur_state = prvFeasible_state
                        ag.dyn.cur_time = prvFeasible_state_time

                        if retrackAbortFlag:
                            pass
                        else:
                            #Move one step and we'll see if the problem is still infeasible:
                            try:
                                ag.step(u=ag.dyn.get_u(), plot=False, time_step=dt)
                                feas_u = True
                            except:
                                feas_u = False
                            if not(feas_u):
                                ag.step(u=np.array([0.5,0]), plot=False, time_step=dt)
                                
                                



            else: # Proceed the next step in time without truncating anything (the feasible case)
                if self.r_dt != self.params.step_size:
                    self.r_dt = self.params.step_size
                [ag.step(u=None, plot=False) for ag in self.agents]
                #Reset the recorded dt as soon it become feasible:
            
            

                # If the next optimization is infeasible retrack otherwise proceed to the next step.
                # A challenge: the moment m.optimize() is infeasible the controls are lost you have to update them manulay till
                # the problem is feasible again!

            #------ original method (no checking)----
            #[ag.step(u=None, plot=False) for ag in self.agents]
            #The ZOH approuch:
            t_now = agt.dyn.time[-1]
            if self.params.plot_sim:
                self.plot_scenario()

            
            if steps is not None:
                for i in range(len(self.agents)):
                    ag = self.agents[i]
                    state = ag.state
                    x_sol[i,0,self.cur_timestep] = state[0,0]
                    x_sol[i,1,self.cur_timestep] = state[1,0]
                    u_sol[i,0,self.cur_timestep] = ag.u[0,0].x
                    u_sol[i,1,self.cur_timestep] = ag.u[1,0].x

            self.cur_timestep += 1

            goal_reached = self.goalReached(retrackAbortFlag)

            if self.params.live_plots:
                self.show_plots()
            self.time_vec.append(self.cur_timestep*self.params.step_size)

            if self.cur_timestep == steps:
                # return (solution[0].x, solution[1].x)
                return x_sol, u_sol

        #Returning the states, and control trajectories:
        t0_ret = timeit.default_timer()
        if self.agents[0].dyn_enum is Dyn.UNICYCLE:
            agentsTrajectory   = []
            agentsU_Trajectory = []
            timeTraj            = []
            for agt in self.agents:
                agentsTrajectory.append(agt.dyn.trajectory)
                agentsU_Trajectory.append(agt.dyn.uTrajectory)
                timeTraj.append(agt.dyn.time)

        elif self.agents[0].dyn_enum is Dyn.UNICYCLE_EXT:
            agentsTrajectory   = []
            agentsU_Trajectory = []
            timeTraj           = []
            for agt in self.agents:
                agentsTrajectory.append(agt.dyn.trajectory)
                agentsU_Trajectory.append(agt.dyn.uTrajectory)
                timeTraj.append(agt.dyn.time)
        elif self.agents[0].dyn_enum is Dyn.UNICYCLE_ACT:
            agentsTrajectory = []
            agentsU_Trajectory = []
            timeTraj = []
            for agt in self.agents:
                agentsTrajectory.append(agt.dyn.trajectory)
                agentsU_Trajectory.append(agt.dyn.uTrajectory)
                timeTraj.append(agt.dyn.time)
        elif self.agents[0].dyn_enum is Dyn.UNICYCLE_DIR:
            agentsTrajectory = []
            agentsU_Trajectory = []
            timeTraj = []
            for agt in self.agents:
                agentsTrajectory.append(agt.dyn.trajectory)
                agentsU_Trajectory.append(agt.dyn.uTrajectory)
                timeTraj.append(agt.dyn.time)
        elif self.agents[0].dyn_enum is Dyn.SINGLE_INT:
            agentsTrajectory = []
            agentsU_Trajectory = []
            timeTraj = []
            for agt in self.agents:
                agentsTrajectory.append(agt.dyn.trajectory)
                # agentsU_Trajectory.append(agt.dyn.uTrajectory)
                timeTraj.append(agt.dyn.time)
        #plotTrajs(agentsTrajectory, agentsU_Trajectory, timeTraj)
        t1_ret = timeit.default_timer()
        T_ret = t1_ret - t0_ret

        return agentsTrajectory, agentsU_Trajectory, timeTraj

'''This function will merely check if the QP, given the current states of the agents/obstacle, is feasible or not.
# gurobi_model.status = {1(loaded), 2(Optimal), 3(Infeasible), ..... } https://www.gurobi.com/documentation/9.1/refman/optimization_status_codes.html
'''
# def get_prev_dt():
#     pass
# def record_pre
def checkFeasibility(simulationObj,m):
    agents = simulationObj.agents
    obsts = simulationObj.obsts
    params = simulationObj.params
    g_model = m
    # Add Pairwise CBF Constraints
    for i in range(len(agents)):
        agt = agents[i]

        if params.decentralized:
            agt2 = range(len(agents))
            agt2.remove(i)
        else:
            agt2 = range(i + 1, len(agents))
        # CBF agent/agent
        for j in agt2:
            simulationObj.add_cbf_pair(g_model, agt, agents[j])
        # CBF agent/obstacle
        for k in range(len(obsts)):
            simulationObj.add_cbf_pair(g_model, agt, obsts[k])
        g_model.update() #Will update the instance not the original gurobi model
        constrs = g_model.getConstrs()
        vars = g_model.getVars()

        g_model.getCoeff(constrs[0],vars[0])

        if g_model.Status is 3: #Infeasible
            FeasFlag = False
        else:
            FeasFlag = True
    return FeasFlag

def make_column(vec):
    vec = np.array(vec)
    vec.shape = (max(vec.shape),1)
    return vec

#this function is to plot the trajectories and the controls:
def plotTrajs(agentsTrajectory, agentsU_Trajectory,timeTraj,trialPrefixName,saveFigs=True):

    #Creat a Figures folder:
    # if not os.path.exists('Figures'):
    #     os.makedirs('Figures')
    # curr_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Figures/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    Nagents = len(agentsTrajectory)
    agt_i   =0 #agent counter
    figXts = [plt for i in range(Nagents)] #The trajectory for each of the states with time
    figXs = [plt for i in range(Nagents)] #The phase plane (the states in the state space)
    figUs = [plt for i in range(Nagents)]
    states_names = ["$x$","$y$","$theta$","$v$"]
    control_names = ["v","\omiga"]
    #States trajectories:
    for agtTrjectory in agentsTrajectory:
        agt_timeTraj = timeTraj[agt_i]
        x_i = 0
        for stateTraj in agtTrjectory:
            figXts[agt_i].plot(agt_timeTraj,stateTraj,label=states_names[x_i])
            x_i+=1
        figName = trialPrefixName + "_agt" + str(agt_i+1)
        #Show the trajectory for each state:
        figXts[agt_i].grid(True)
        figXts[agt_i].legend()
        if saveFigs:
            figXts[agt_i].savefig(results_dir + figName + "_stateTraj")
        figXts[agt_i].show()
        #Plot the phase plane:
        figXs[agt_i].plot(agtTrjectory[0,:],agtTrjectory[1,:])
        figXs[agt_i].xlabel('$x(t)$')
        figXs[agt_i].ylabel('$y(t)$')
        if saveFigs:
            figXts[agt_i].savefig(results_dir + figName + "_statesPlane")
        figXs[agt_i].show()

        agt_i +=1
    #Controls trajectories:
    agt_i = 0
    for agtU_Trjectory in agentsU_Trajectory:
        #Figure setting:
        figName = trialPrefixName + "_agt" + str(agt_i + 1)

        agt_timeTraj = timeTraj[agt_i]
        u_i = 0
        for uTraj in agtU_Trjectory:
            figUs[agt_i].plot(agt_timeTraj,uTraj,label=control_names[u_i])
            u_i+=1
        figUs[agt_i].grid(True)
        figUs[agt_i].legend()
        if saveFigs:
            figXts[agt_i].savefig(results_dir + figName + "_CtrlTraj")
        figUs[agt_i].show()
        agt_i +=1
def arcLength(qTrajectory):
    qTrajectory = np.asarray(qTrajectory)
    x = qTrajectory[:,0]
    y = qTrajectory[:,1]
    arc_Length = 0

    for i in range(len(qTrajectory[:,0])):
        if i==0:
            arc_Length = 0
        else:
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            darcLength = math.sqrt(dx**2+dy**2)
            arc_Length = arc_Length + darcLength

    return arc_Length


def main():

    Params('configs.ini')
    sim = Simulation()

    # a = 8
    # u_ref = np.hstack((np.ones((2,a)), np.zeros((2,100-a)))) * 1
    # sim.add_agent(Agent((0,0),u_ref,dynamics=Dyn.DOUBLE_INT))
    # sim.add_obstacle(Sphere((3,3.9), 1, dynamics=Dyn.DOUBLE_INT))

    # sim.add_agent((0, 0), goal=(20,0), radius=1, dynamics=Dyn.SINGLE_INT)


    #dyn = Dyn.UNICYCLE_EXT
    dyn = Dyn.UNICYCLE
    #dyn = Dyn.UNICYCLE_ACT
    #dyn = Dyn.UNICYCLE_DIR
    #dyn = Dyn.SINGLE_INT
    ############## Unicycle Extended dynamics:
    flagGoal = 0
    if dyn is Dyn.UNICYCLE_EXT:
        if flagGoal:
            sim.add_agent(Agent((0, 0), Goal((10, 0)), dynamics=dyn))
            #sim.add_obstacle(Sphere([15, 0], radius=1))
            #sim.add_agent(Agent((10, 1), Goal((0, 0)), dynamics=dyn))

        else:

            wRef  = np.linspace(0,0,20)
            #muRef = np.append(np.linspace(.51, .51, 50),[np.linspace(0,0, 50),np.linspace(-.51,-.51, 50)])
            muRef = np.linspace(1.0/10,1.0/10,20)
            #muRef = np.append(np.linspace(1.0 /5, 1.0 /5, 100), np.zeros([1, 150]))

            u_ref = np.vstack([wRef,muRef])
            sim.add_agent(Agent(np.vstack([0,0]), u_ref,radius=.5, dynamics=dyn))
            #sim.add_agent(Agent((10, 0),-u_ref, dynamics=dyn))
            #sim.add_obstacle(Ellipsoid([1,1], [1, .250], angle=70))
            #sim.add_obstacle(Ellipsoid([1,1], [1, .250], angle=-70))
            # sim.add_obstacle(Sphere([5.0,.09],radius=.5))  #below 0.007 the optimization problem becomes infeasible,
            #                                                  # TODO: Return error when the optimization problem become infeasible!!
            # sim.add_obstacle(Sphere([2, .7], radius=.5))
            # sim.add_obstacle(Sphere([5, -1], radius=.5))
            # sim.add_obstacle(Sphere([10.5, -1], radius=.5))
            # sim.add_obstacle(Sphere([13, 0], radius=1))
            #
            # sim.add_obstacle(Sphere([1, 0], radius=.2))




    ############ Unicycle integrator and mapping
    elif dyn is Dyn.UNICYCLE:
        if flagGoal:
            vRef = np.linspace(1.0, 1.0, 1500)
            sim.add_agent(Agent((0, 0), Goal((10, 0),u_ref=vRef), dynamics=dyn))
            sim.add_obstacle(Sphere([5, 0.5], radius=.5))
            #sim.add_agent(Agent((10, 0), Goal((0, 0),u_ref=vRef), dynamics=dyn))
            # sim.add_agent(Agent((0,-10), Goal((0, 10), u_ref=vRef), dynamics=dyn))
            # sim.add_agent(Agent((0, 10), Goal((0, -10), u_ref=vRef), dynamics=dyn))

            #sim.add_agent(Agent((5, 10), Goal((5,-10)), dynamics=dyn))
            #sim.add_agent(Agent((5,-10), Goal((5,10)), dynamics=dyn))
        else:
            desired_theta = math.atan2(-1, -1)
            vRef = np.linspace(1.0,1.0, 100)
            wRef = np.linspace(-45/57.3,-45/57.3,100)
            u_ref = np.array([vRef, wRef])
            sim.add_agent(Agent((0, 0),theta=-45/57.3,instructs=u_ref, dynamics=dyn))
            #sim.add_agent(Agent((10.0, 0),-u_ref, dynamics=dyn))
            # sim.add_obstacle(Ellipsoid([6.0, 1.5], [.3, 2], angle=45))
            # sim.add_obstacle(Ellipsoid([6.0, 1.5], [.3, 2], angle=-45))
            #sim.add_obstacle(Sphere([5, 1.2], radius=.5))
            # sim.add_obstacle(Sphere([10, 0], radius=.5))
            # sim.add_obstacle(Sphere([18, 0], radius=.5))

    ########## Unicycle actuator dynamics:
    elif dyn is Dyn.UNICYCLE_ACT:
        if flagGoal:
            sim.add_agent(Agent((0, 0), Goal((30, 0)), dynamics=dyn))
            #sim.add_obstacle(Sphere([15, 0], radius=1))
            sim.add_agent(Agent((30, 0), Goal((0, 0)), dynamics=dyn))
            # sim.add_agent(Agent((5, 10), Goal((5,-10)), dynamics=dyn))
            # sim.add_agent(Agent((5,-10), Goal((5,10)), dynamics=dyn))
        else:
            slRef = np.linspace(0, 1, 500)
            srRef = np.linspace(0, 1, 500)
            u_ref = np.vstack([slRef, srRef])
            sim.add_agent(Agent((0, 1.001), u_ref, dynamics=dyn))
            #sim.add_agent(Agent((10.0, 0),-u_ref, dynamics=dyn))
            sim.add_obstacle(Sphere([12, 0], radius=.5))
    ################# Unicycle direct dynamics
    elif dyn is Dyn.UNICYCLE_DIR:
        if flagGoal:
            sim.add_agent(Agent((0, 0), Goal((30, 0)), dynamics=dyn))
            # sim.add_obstacle(Sphere([15, 0], radius=1))
            sim.add_agent(Agent((30, 0), Goal((0, 0)), dynamics=dyn))
            # sim.add_agent(Agent((5, 10), Goal((5,-10)), dynamics=dyn))
            # sim.add_agent(Agent((5,-10), Goal((5,10)), dynamics=dyn))

        else:
            vRef = np.linspace(0, 1, 200)
            wRef = np.linspace(0, 0, 200)
            u_ref = np.vstack([vRef,wRef])
            sim.add_agent(Agent((0.0,1.1), u_ref, dynamics=dyn))
            #sim.add_agent(Agent((10.0,0.0),-u_ref, dynamics=dyn))
            sim.add_obstacle(Sphere([5,0.95],radius=.5))
    elif dyn is Dyn.SINGLE_INT:
        if flagGoal:
            vRef = np.linspace(1.0, 1.0, 1500)
            sim.add_agent(Agent((0, 0), Goal((10, 0),u_ref=vRef), dynamics=dyn))
            sim.add_obstacle(Sphere([5, 0.5], radius=.5))
            #sim.add_agent(Agent((10, 0), Goal((0, 0),u_ref=vRef), dynamics=dyn))
            # sim.add_agent(Agent((0,-10), Goal((0, 10), u_ref=vRef), dynamics=dyn))
            # sim.add_agent(Agent((0, 10), Goal((0, -10), u_ref=vRef), dynamics=dyn))

            #sim.add_agent(Agent((5, 10), Goal((5,-10)), dynamics=dyn))
            #sim.add_agent(Agent((5,-10), Goal((5,10)), dynamics=dyn))
        else:
            desired_theta = math.atan2(-1, -1)
            vRef = np.linspace(1.0,1.0, 15)
            wRef = np.linspace(0,0, 15)
            u_ref = np.vstack([vRef, wRef])
            sim.add_agent(Agent((0, 0),u_ref, dynamics=dyn))
            #sim.add_agent(Agent((10.0, 0),-u_ref, dynamics=dyn))
            sim.add_obstacle(Ellipsoid([5.0, 0], [.3, 2], angle=45))
            sim.add_obstacle(Ellipsoid([5.0, 0], [.3, 2], angle=-45))
            # sim.add_obstacle(Sphere([5, 0], radius=.5))
            # sim.add_obstacle(Sphere([10, 0], radius=.5))
            # sim.add_obstacle(Sphere([18, 0], radius=.5))

    #sim.add_agent(Agent((2,2),Goal((-4,8)),dynamics=dyn), priorityVal=0.25)

    #sim.add_obstacle(Wall((0,20),(5,10), np.array([[1],[.4]]), -10))
    
    
    #sim.add_obstacle(Sphere((5.1,0), 0.5))

    agentsTrajectory, agentsU_Trajectory, timeTraj = sim.initiate()

    dumlen = arcLength(agentsTrajectory[0])


    trialPrefixName = 'ExtU_ref_p1p1_muCons_crossObs1_-,1'
    plotTrajs(agentsTrajectory, agentsU_Trajectory, timeTraj, trialPrefixName,saveFigs=False)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime:",end_time-start_time)