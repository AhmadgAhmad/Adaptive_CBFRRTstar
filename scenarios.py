import numpy as np
from simulation import Simulation
from dynamics import Dyn
from obstacle import Sphere, Ellipsoid, Wall
from agent import Agent
from goal import Goal
from params import *

def one_agent_one_obstacle(sim, dyn=Dyn.SINGLE_INT, agent_rad=0.5, obst_rad=1.0):
    sim.add_agent(Agent((0,0), Goal((10,0)),radius=agent_rad))
    sim.add_obstacle(Sphere((5,0.1), obst_rad))

def two_agents(sim, dyn=Dyn.UNICYCLE):
    sim.add_agent(Agent((0,0), Goal((5,0)), dynamics=dyn))
    sim.add_agent(Agent((5,0), Goal((0,0)), dynamics=dyn,theta=np.pi))

def scen_circle_of_agents(sim, numAgents, circleRad=5, agentRad=0.5, dynamics=Dyn.SINGLE_INT):
    d_angle = int(360/numAgents)
    
    # for angle_deg in list(range(0, 360, d_angle)):
    for i in range(numAgents):
        angle = np.radians(i*d_angle)
        x0 = np.cos(angle) * circleRad + (np.random.uniform(-0.05,0.05))
        y0 = np.sin(angle) * circleRad + (np.random.uniform(-0.05,0.05))
        x1 = -x0
        y1 = -y0
        sim.add_agent(Agent((x0,y0),Goal((x1,y1)),radius=agentRad,dynamics=dynamics,theta=angle+np.pi + (np.random.uniform(0,0.05))))

def circle_agent_ellipses(sim, numAgents, circleRad=4.0, agentRad=0.5, dynamics=Dyn.SINGLE_INT):
    angle_diff = 2*np.pi / numAgents

    # count = 0
    # cur_angle = 0
    # while count < numAgents:
    #     x = np.rint(np.cos(cur_angle) * circleRad * 100) / 100
    #     y = np.rint(np.sin(cur_angle) * circleRad * 100) / 100
    #     init = (x+circleRad+1,y+circleRad+1)
    #     goal = (-x+circleRad+1,-y+circleRad+1)
    #     sim.add_agent(Agent(init,Goal(goal),radius=agentRad,dynamics=dynamics))
    #     cur_angle += angle_diff
    #     count += 1

    init = (2,5)
    goal = (7,5)
    sim.add_agent(Agent(init,Goal(goal),radius=agentRad,dynamics=dynamics))

    center = (circleRad+1,circleRad+1)
    if numAgents == 1 or numAgents == 2 or numAgents == 4:
        sim.add_obstacle(Ellipsoid(center, (circleRad/4, 0.1), 45))
        sim.add_obstacle(Ellipsoid(center, (circleRad/4, 0.1), -45))
    else: # ODD
        pass

def testing(sim,num):
    sim.add_agent_new(Agent((0,0), Goal((0,5))))
    sim.add_agent_new(Agent((0,5), Goal((0,0))))
    sim.add_agent_new(Agent((1,2), Goal((-1,-2))))

if __name__ == '__main__':
    Params('configs.ini')
    sim = Simulation()
    # circle_agent_ellipses(sim, 1)
    # scen_circle_of_agents(sim, 4, dynamics=Dyn.UNICYCLE, circleRad=2, agentRad=0.17)
    scen_circle_of_agents(sim, 4, dynamics=Dyn.UNICYCLE, circleRad=3)
    # two_agents(sim)
    sim.initiate()