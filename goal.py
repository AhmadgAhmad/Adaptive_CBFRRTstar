import numpy as np
from dynamics import SingleIntegrator, UnicycleActuators,Unicycle, \
               DoubleIntegrator, UnicycleExtended, UnicycleDirect
from function import Function
from gurobipy import *
from params import *
import matplotlib.pyplot as plt
from params import Params


class Goal(object):

    def __init__(self, pos,u_ref = None):
        self.params = Params()
        self.goal = np.array(pos)
        self.v = Function(self.v_func, self.del_v)
        self.u_ref = u_ref #This must be fed for the inputs that is not a control variable (it is a vector, i.e. typically
                           #if one to choose to have w to be the control and v/mu constant (u_ref) for the unucycle, or extended respectively)

    def v_func(self, agt):
         # if not isinstance(agt.dyn, DoubleIntegrator):
        #     return x.T.dot(x)/2
        # else:
        #     return x.T.dot(x)/2  + np.linalg.norm(agt.state[2:])**2 /2
        if not isinstance(agt.dyn, UnicycleExtended):
            x = agt.state[0:2] - self.goal
            return x.T.dot(x)/2
        else:
            q_goal = np.vstack([self.goal])
            x = agt.state[0:2] - q_goal
            return x.T.dot(x)/2 # + np.linalg.norm(agt.state[0:1])**2 /2

    def del_v(self, agt):
        # if not isinstance(agt.dyn, DoubleIntegrator):
        #     return agt.state - self.goal
        # else:
        #     return agt.state - self.goal + np.linalg.norm(agt.state[2:])
        if not isinstance(agt.dyn, UnicycleExtended):
            return agt.state[0:2] - self.goal
        else:
            q_goal = np.vstack([self.goal])
            return agt.state[0:2] - q_goal# + np.linalg.norm(agt.state[0:2])

    def distance_to_goal(self, agt):
        return np.sqrt((agt.state[0:2]-self.goal).T.dot(agt.state[0:2]-self.goal)[0][0])

    def plot(self, ax=None, color='g'):
        if ax is None:
            _, ax = plt.gca()
        ax.plot(self.goal[0], self.goal[1], 'x', color=color, markersize=10, markeredgewidth=5)
        ax.axis('equal')

    def __repr__(self):
        return "Goal({},{})".format(self.goal[0,0], self.goal[1,0])

    def add_clf(self, m, agt):
        params = Params()
        if type(agt.dyn) is SingleIntegrator:
            H = np.identity(len(agt.u)) * .25
            p = .25
            gamma = 4
        elif type(agt.dyn) is Unicycle:    
            # TODO: MAKE PARAMS TAKE A PARAM FILE. USE CONFIG. SEPARATE PARAMS
            vel_penalty =  params.vel_penalty #2 # was 10
            steer_penalty = params.steer_penalty # 1
            H = np.array([[vel_penalty, 0], [0, steer_penalty]])
            p = params.p
            gamma = params.gamma
        elif type(agt.dyn) is DoubleIntegrator:
            H = np.identity(len(agt.u)) * 10
            p = 20
            gamma = .25
        elif type(agt.dyn) is UnicycleExtended:
            vel_penalty = params.vel_penalty  # 2 # was 10
            steer_penalty = params.steer_penalty  # 1
            H = np.identity(2)
            p = params.p
            gamma = params.gamma
        elif type(agt.dyn) is UnicycleActuators:
            s_penalty = params.s_penalty  # 2 # was 10
            H = np.array([[s_penalty, 0], [0, s_penalty]])
            p = params.p
            gamma = params.gamma
        elif type(agt.dyn) is UnicycleDirect:
            vel_penalty = params.vel_penalty  # 2 # was 10
            steer_penalty = params.steer_penalty  # 1
            H = np.array([[vel_penalty, 0], [0, steer_penalty]])
            p = params.p
            gamma = params.gamma
        # Add relaxation variable delta
        delta = m.addVar(vtype=GRB.CONTINUOUS, name="delta{}".format(agt.id))
        v_val = self.v.eval(agt)
        lf_v = (self.v.grad(agt).T.dot(agt.get_q_dot()[0:2]))
        # lf_v = gamma*self.v.grad(agt).transpose().dot(H).dot(agt.get_x_dot())[0][0]
        #lf_v.getVar(1)
        #constraint = (lf_v + gamma * v_val <= delta)
        constraint = (delta<=lf_v + gamma*v_val)
        # cost_func = 0.5 * agt.u.T.dot(np.identity(2)).dot(agt.u)[0][0]+\
        #             p*delta*delta #need to replace the identity weight

        cost_func = 0.5 * agt.u.T.dot(np.identity(2)).dot(agt.u) + \
                    p * delta * delta + gamma*self.v.grad(agt).T.dot(H).dot(agt.get_q_dot()[0:2])
            #
        if type(agt.dyn) is not Unicycle:
            m.addConstr(constraint)
        m.setObjective(cost_func + m.getObjective(), GRB.MINIMIZE)
        m.update()

        return v_val