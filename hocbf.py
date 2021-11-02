
import numpy as np
from function import Function
from obstacle import Sphere, Ellipsoid, Wall
from dynamics import Dyn
from enum import Enum
from gurobipy import *
from params import *
import timeit

class typeAlpha(Enum):
    LIN_LIN = 0      #Linear - Linear
    QUAD_QUAD = 1    #Quadratic - Linear
    QUAD_LIN = 2     #Quadratic - Linear
    LIN_QUAD = 3    #Linear - Quadratic

    #SQR_ROOT = 1

class HOCBF(object):
    def __init__(self,simObject,m,agt,obst,rDegree = 1, type_alpha = typeAlpha.LIN_LIN):
        self.params = Params()
        self.rDegree = rDegree
        self.type_alpha = type_alpha
        self.agt = agt
        self.obst = obst
        self.m = m
        self.simObject = simObject

    def get_alpha1(self):
        agt = self.agt
        obst = self.obst
        type_alpha = self.type_alpha


        if type_alpha==typeAlpha.LIN_LIN or type_alpha==typeAlpha.LIN_QUAD:
            #Linear alpha1(h)
            p1 = self.params.alpha_linear_p1
            alpha1_h_q = p1*obst.h.eval(agt)
        elif type_alpha==typeAlpha.QUAD_LIN or type_alpha==typeAlpha.QUAD_QUAD:
            #Quadratic alpha1(h)
            alpha1_h_q = obst.h.eval(agt)**2
        return alpha1_h_q

    def get_alpha2(self):
        type_alpha = self.type_alpha
        psi1 = self.get_psi1()

        if type_alpha==typeAlpha.LIN_LIN or type_alpha==typeAlpha.QUAD_LIN:
            #Linear alpha2(psi1)
            p2 = self.params.alpha_linear_p2
            alpha2_psi1 = p2*psi1
        elif type_alpha==typeAlpha.LIN_QUAD or type_alpha==typeAlpha.QUAD_QUAD:
            #quadratic alpha2(psi1)
            alpha2_psi1 = psi1**2

        return alpha2_psi1

    def get_psi1(self):
        agt = self.agt
        obst = self.obst
        q_dot = agt.get_x_dot()[0:2] - obst.get_x_dot()[0:2]
        alpha1_h = self.get_alpha1()
        # psi1 = psi0 + alpha1(psi0); psi0 = h_q:
        psi1 = obst.h.grad(agt).T.dot(q_dot[0:2]) + alpha1_h
        return psi1

    def get_psi2(self):
        agt = self.agt
        obst = self.obst
        q_dot = agt.get_x_dot()[0:2] - obst.get_x_dot()[0:2]

        if isinstance(obst, Sphere): #This means static obstacle
            obst_x_dot_dot = np.vstack([0, 0])
        elif isinstance(obst, Ellipsoid):
            obst_x_dot_dot = np.vstack([0, 0])
        else: #Here, however, it is a spherical obstacle but an agent. (TODO: it is basically a bug)
            obst_x_dot_dot = obst.get_x_dot(n=2)
        h_hess = np.vstack([[2, 0], [0, 2]])

        x_dotdot = agt.get_x_dot(n=2)[0] - obst_x_dot_dot[0]
        y_dotdot = agt.get_x_dot(n=2)[1] - obst_x_dot_dot[1]
        xy_dotdot = np.vstack([x_dotdot, y_dotdot])
        psi1_dot = obst.h.grad(agt).T.dot(xy_dotdot) + q_dot[0:2].T.dot(h_hess).dot(q_dot[0:2])
        alpha2_psi1 = self.get_alpha2()

        psi2 = psi1_dot + alpha2_psi1

        return psi2 #It is needed to be returned if you'd define higher orders

    #do it outside here since I'm alraedy just using the 1st order CBF.
    def Update_m(self):
        m = self.m

        m.getConstrs()

        # obst = self.obst

        agt = self.agt
        simObject = self.simObject
        rDegree = self.rDegree

        if rDegree == 1:
            t0_updtm = timeit.default_timer()
            psi1 = self.get_psi1()
            t1_updtm = timeit.default_timer()
            T_updtm = t1_updtm - t0_updtm

            constr = m.addConstr((psi1[0,0]>=0.0), name="CBF_{}".format(agt.id))

            attr = GRB.Attr.RHS
            m.update()
        elif rDegree == 2:
            psi2 = self.get_psi2()
            constr = m.addConstr((psi2[0,0] >= 0), name="CBF_{}".format(agt.id))
            attr = GRB.Attr.RHS
            m.update()
        # Check if you need HOCBF
        #psi2
        #dummy =  self.get_hocbf_val()
        m.update()

    def get_hocbf_val(self):
        agt = self.agt
        obst = self.obst
        h_val = obst.h.eval(agt)
        return h_val

    def feasibilityCheck(self):
        m = self.m #The gurobi model
        FlagInfeas = False
        if m.Status==3:
            FlagInfeas = True
        return FlagInfeas

    #TODO: work on getting the constraint value if you want to plot it.
    def get_GRB_constr(self):
        m = self.m
        constr = m.getAttr()


def main():
    pass

main()

