"""
Author: Ahmad Ahmad, ahmadgh@bu.edu
This class is define CBF-QP, and CLF-CBF-QP local trajectory planner for (Adaptive) CBF-RRT* algorithm 
"""
import numpy as np 
import agent,goal,dynamics, simulation

class Steer:
    def __init__(self, v_init, v_f, obsList = None, m=10, simObject = None):
        
        self.v_init = v_init        # The initial vertex
        self.v_f = v_f              # The final vertex
        self.m = m                  # The number of discrete steps of the QP controller 
        #self.extFlag = extFlag      # Enabling flag for exact steering (if enabled, CLF constrint will be included ) 
        self.obsList = obsList      # The list of obstacles
        self.simObject = simObject  # Simulation object that contains the numerical methods to evolve the system dynamics, ... ets 
    
    def safeSteer(self, v_init, v_f, m = 10):
        if m<1:
            m=1
        vRef = np.linspace(1.0, 1.0, m)  # Assume we're working with the unicycle dynamics
        wRef = np.linspace(desired_theta, desired_theta, m)
        if len(wRef) is 0:
            raise Exception('check out the desired theta')
        u_ref = np.vstack([vRef, wRef]) #TODO (slow)
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
        self.simObject.add_agent(Agent(xy_v_nearest, radius=.4,theta=desired_theta ,instructs=u_ref, dynamics=Dyn.UNICYCLE))
        qTrajectory, uTrajectory, tTrajectory = self.simObject.initiate()  # TODO: incorporate \theta with the trajectory
        qFinal = qTrajectory[0][:, -1]
        tFinal = tTrajectory[0][-1] + tInitial
        return qFinal, tFinal, uTrajectory[0], qTrajectory[0], tTrajectory[0] + tInitial
    def ext_SafeSteer(self,v_init, v_f):
        pass