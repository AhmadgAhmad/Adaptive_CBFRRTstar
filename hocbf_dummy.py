"""""
This script is written by Ahmad Ahmad 2-6-2021. 

The HOCBF class will update the Guroby model with the HOCBF, or CBF if the relative degree is one. 
The class attributes are: agent, obstacle, the gurobi model, the relative degree of the system, and 
type of the class-k function that is used to define the HOCBF [1]. 

The class has a method that will update the gurobi model with the HOCBF constraint. It will  also 
return the constraint value if needed to plot it. 
 
"""""


import numpy as np
from function import Function
from obstacle import Sphere, Ellipsoid, Wall
from dynamics import Dyn
from enum import Enum
from gurobipy import *

class typeAlpha(Enum):
    LINEAR = 0
    SQR_ROOT = 1
    QUADRATIC = 2

class HOCBF(object):
    def __init__(self,simObject,m,agt,obst,rDegree = 1, type_alpha = typeAlpha.LINEAR):
        self.rDegree = rDegree
        self.type_alpha = type_alpha
        self.agt = agt
        self.obst = obst
        self.m = m
        self.simObject = simObject

    def Update_m(self):
        m = self.m
        obst = self.obst
        agt = self.agt
        simObject = self.simObject

        if simObject.params.decentralized:
            obst_x_dot = obst.get_x_dot((0, 0))
            k_cbf = obst.k_cbf
        else:
            obst_x_dot = obst.get_x_dot()
            k_cbf = 1.0
        p_cbf = simObject.params.p_cbf

        h_val = obst.h.eval(agt)
        # lg_h = obst.h.grad(agt).T.dot(agt.get_x_dot() - obst_x_dot)[0][0]
        h_hess = np.vstack([[2, 0], [0, 2]])
        # Get the 2nd derivative of x and y of the agent
        # agt.dyn_enum = Dyn.UNICYCLE_EXT
        if agt.dyn_enum is Dyn.UNICYCLE_EXT:
            k1_cbf = 1.0
            q = (agt.state[0:2] - obst.state[0:2])
            q_dot = agt.get_x_dot()[0:2] - obst.get_x_dot()[0:2]
            if isinstance(obst, Sphere):
                obst_x_dot_dot = np.vstack([0, 0])
            elif isinstance(obst, Ellipsoid):
                obst_x_dot_dot = np.vstack([0, 0])
            else:
                obst_x_dot_dot = obst.get_x_dot(n=2)

            x_dotdot = agt.get_x_dot(n=2)[0] - obst_x_dot_dot[0]
            y_dotdot = agt.get_x_dot(n=2)[1] - obst_x_dot_dot[1]
            xy_dotdot = np.vstack([x_dotdot, y_dotdot])
            # HOCBF (2nd-order with linear class-k alphas)
            # psi2 = h_dd +(p1+p2)h_d + p1p2h
            p1, p2 = .1, .1
            psi2 = obst.h.grad(agt).T.dot(xy_dotdot) + q_dot[0:2].T.dot(h_hess).dot(q_dot[0:2]) \
                   + (p2) * obst.h.grad(agt).T.dot(q_dot[0:2]) + (p1 * p2) * obst.h.eval(agt)
            # The HOCBF constraint:
            #constr = m.addConstr((q_dot[0:2].T.dot(h_hess).dot(q_dot[0:2]) \
            #                      + (p1 + p2) * obst.h.grad(agt).T.dot(q_dot[0:2]) + obst.h.grad(agt).T.dot(xy_dotdot))[
            #                         0, 0] >= -(p1 * p2) * obst.h.eval(agt), name="CBF_{}".format(agt.id))
            constr = m.addConstr((psi2[0, 0] >= 0), name="CBF_{}".format(agt.id))
            attr = GRB.Attr.RHS

        # Check if you need HOCBF
        elif agt.dyn_enum is Dyn.DOUBLE_INT:
            # Could check lg_h to see if u shows up using lg_h.getVar()
            # and lg_h.getCoeff() but that may be overkill for now
            k1_cbf = 1.0
            x = (agt.state - obst.state)
            x_dot = agt.get_x_dot() - obst.get_x_dot()
            x_ddot = np.vstack((x_dot[2:], 0, 0))
            lg2_h = 2 * (x_dot.T).dot(obst.M).dot(x_dot) + 2 * (x.T).dot(obst.M).dot(x_ddot)

            constr = m.addQConstr((lg2_h + k_cbf * lg_h + k1_cbf * (lg_h + k_cbf * h_val))[0][0] >= 0,
                                  name="CBF_{}".format(agt.id))
            attr = GRB.Attr.QCRHS
        else:
            # constr = m.addConstr((lg_h)>=-k_cbf*h_val**p_cbf, name="CBF_{}".format(agt.id))
            # constraint
            h_val = obst.h.eval(agt)
            lg_h = obst.h.grad(agt).T.dot(agt.get_x_dot()[0:2] - obst_x_dot[0:2])[0][0]
            constr = m.addConstr((lg_h) >= - k_cbf * h_val ** p_cbf, name="CBF_{}".format(agt.id))
            attr = GRB.Attr.RHS
            m.getVars()
        m.update()
        m.getVars()

    def get_hocbf_val(self):
        agt = self.agt
        h_val = agt.h.eval(agt)
        return h_val

    #TODO: work on getting the constraint value if you want to plot it.
    def get_GRB_constr(self):
        m = self.m
        constr = m.getAttr()


def main():
    pass

main()

