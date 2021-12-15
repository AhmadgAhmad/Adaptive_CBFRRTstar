"""
Author: Ahmad Ahmad, ahmadgh@bu.edu
This class is define CBF-QP, and CLF-CBF-QP local trajectory planner for (Adaptive) CBF-RRT* algorithm 
"""
import numpy as np 
import agent,goal,dynamics, simulation

class Safe_steer:
    
    def __init__(self, v_init, v_f, obsList = None, m=10, extFlag = False, simObject = None):
        
        self.v_init = v_init        # The initial vertex
        self.v_f = v_f              # The final vertex
        self.m = m                  # The number of discrete steps of the QP controller 
        self.extFlag = extFlag      # Enabling flag for exact steering (if enabled, CLF constrint will be included ) 
        self.obsList = obsList      # The list of obstacles
        self.simObject = simObject  # Simulation object that contains the numerical methods to evolve the system dynamics, ... ets 

    def steer(self):
        """
        steers the robot state form the configuration of v_init to the configuration of v_f
        inputs:
        outputs: 
        """
        
        if self.extFlag: 
            #TODO [RSS] define a CLF-CBF constrint 
            pass
        else:
            #TODO [RSS] define a CBF-constraints  
            pass 
        pass

    def exactSteer(self):
        pass

    def safeSteer(self):
        pass

