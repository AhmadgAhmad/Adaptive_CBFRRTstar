import numpy as np
import math
import matplotlib.pyplot as plt

from gurobipy import *
from params import *
from function import Function
from dynamics import SingleIntegrator, UnicycleActuators,Unicycle, \
                    DoubleIntegrator, Dyn, UnicycleExtended,UnicycleDirect
import abc
from agent import Agent

class Obstacle(object):

    @abc.abstractmethod
    def __init__(self, loc):

        self.dyn = SingleIntegrator(loc) # TODO: Need to add a step function for moving obst. Accept different types of dynamics
        self.state = self.dyn.get_state()
        self.h = Function(self.h_func, self.h_func_grad, self.h_func_hess)

    @abc.abstractmethod
    def h_func(self,a1):
        pass

    @abc.abstractmethod
    def h_func_grad(self, a1):
        pass

    @abc.abstractmethod
    def h_func_hess(self,a1):
        pass

    @abc.abstractmethod
    def get_x_dot(self):
        return self.dyn.get_x_dot(self.state)

    @abc.abstractmethod
    def get_x_dot_dot(self):
        return self.dyn.get_x_dot_dot(self.state)

    @abc.abstractmethod
    def get_state(self):
        return self.dyn.cur_state()

    @abc.abstractmethod
    def plot(self):
        pass

    def plot_levelsets(self, ax=None):
        self.h.plot_levelsets(self.state, ax)

class Sphere(Obstacle):
    def __init__(self, loc, radius=1, dynamics=Dyn.SINGLE_INT, hollow=False, k_cbf=1.0):
        self.loc = loc
        self.radius = radius
        if dynamics is Dyn.SINGLE_INT:
            self.dyn = SingleIntegrator(loc)
            self.M = np.identity(2)

        elif dynamics is Dyn.UNICYCLE:
            self.dyn = Unicycle(loc)
            self.M = np.identity(2)

        elif dynamics is Dyn.UNICYCLE_EXT:
            self.dyn = UnicycleExtended(loc)
            self.M = np.identity(2)

        elif dynamics is Dyn.UNICYCLE_ACT:
            self.dyn = UnicycleActuators(loc)
            self.M = np.identity(2)

        elif dynamics is Dyn.DOUBLE_INT:
            self.dyn = DoubleIntegrator(loc)
            self.M = np.diag((1,1,0,0))

        elif dynamics is Dyn.UNICYCLE_DIR:
            self.dyn = UnicycleDirect(loc)
            self.M = np.identity(2)

        self.state = self.dyn.cur_state
        self.h = Function(self.h_func, self.h_func_grad, self.h_func_hess)
        self.sign = -1 if hollow else 1
        self.k_cbf = k_cbf

        # self.grad_test()

    #Here the barrier function
    def h_func(self, a1):
        if isinstance(self.dyn,UnicycleExtended):
            hStates    = self.state[0:2]
            hStates_a1 = a1.state[0:2]
        else:
            hStates = self.state[0:2]
            hStates_a1 = a1.state[0:2]
        return ((hStates_a1-hStates).T.dot(self.M).dot((hStates_a1-hStates))[0][0] - (self.sign*a1.radius+self.radius)**2) * self.sign
        
    def h_func_grad(self, a1):
        if isinstance(self.dyn, UnicycleExtended):
            hStates = self.state[0:2]
            hStates_a1 = a1.state[0:2]
        else:
            hStates = self.state[0:2]
            hStates_a1 = a1.state[0:2]
        return (2*self.M.dot((hStates_a1-hStates))) * self.sign

    def h_func_hess(self, a1):
        return 2*self.M

    def get_state(self):
        return super(Sphere).get_state()

    def get_x_dot(self, u=None):
        return self.dyn.get_x_dot(self.state)

    def get_x_dot_dot(self):
        return self.dyn.get_x_dot_dot(self.state)

    def plot(self, ax=None):
        plot_ellipse(self.loc, self.radius, ax=ax)

    def grad_test(self):
        #define line (curve) along which to evaluate derivative
        x0=np.random.standard_normal(2)
        v0=np.random.standard_normal(2)
        a0=np.random.standard_normal(2)

        x0 = np.array([ [x0[0]], [x0[1]], [v0[0]], [v0[1]] ])
        v0 = np.array([ [v0[0]], [v0[1]], [a0[0]], [a0[0]] ])
    
        x=lambda t: x0+t*v0
        
        #define test points
        nb_points=100
        t_eval=np.linspace(0,1,nb_points)
        delta=1e-8

        #compare analytical and numerical derivatives
        func_val=np.zeros([nb_points, 1])
        der_eval=np.zeros([nb_points, 1])
        der_expected=np.zeros([nb_points, 1])
        
        for idx in range(0,nb_points):
            x_eval=x(t_eval[idx])
            a1 = Agent(x_eval[0:2], init_vel=x_eval[2:4], dynamics=Dyn.DOUBLE_INT)
            func_val[idx] = self.h_func(a1)
            der_eval[idx] = self.h_func_grad(a1).transpose().dot(v0)
            
            a2_state = x(t_eval[idx]+delta)
            a2 = Agent(a2_state[0:2], init_vel=a2_state[2:4], dynamics=Dyn.DOUBLE_INT)
            der_expected[idx]=(self.h_func(a2)-self.h_func(a1))/delta
            

        #plot der_eval, der_expected vs t_eval
        plt.clf()
        plt.subplot()
        plt.plot(t_eval, func_val)
        plt.plot(t_eval, der_eval, linewidth=2)
        plt.plot(t_eval, der_expected, '--', linewidth=2)
        plt.legend(('func', 'der_eval', 'der_expected'))
        plt.show()

class Ellipsoid(Obstacle):
    # TODO: Generalize this for higher dimensionality?
    def __init__(self, loc, axis_lengths, angle=0, Dyn_Class=SingleIntegrator, hollow=False):
        self.loc = loc
        self.a = axis_lengths[0] # Axis length along the x-axis
        self.b = axis_lengths[1] # Axis length along the y-axis
        self.dyn = Dyn_Class(loc)
        self.state = self.dyn.cur_state
        self.angle = angle # DEGREES
        self.sign = -1 if hollow else 1
        self.h = Function(self.h_func, self.h_func_grad)
        self.k_cbf = 1.0

        #Computing the matrix of the ellipsoid:
        a_aug = self.a + ((0.1 * self.sign) * -1)
        b_aug = self.b + ((0.1 * self.sign) * -1)
        theta1 = math.radians(angle)

        c = math.cos(theta1)
        s = math.sin(theta1)

        aa = (c / a_aug) ** 2 + (s / b_aug) ** 2
        bb = s * c * ((1 / b_aug) ** 2 - (1 / a_aug) ** 2)
        cc = (s / a_aug) ** 2 + (c / b_aug) ** 2
        M_ellip = np.array([[aa, bb], [bb, cc]])
        self.M_ellip = M_ellip

    def h_func(self, a1):
        return ((a1.state[0:2]-self.state[0:2]).transpose().dot(self.get_M(a1)).dot((a1.state[0:2]-self.state[0:2]))[0][0] - 1) * self.sign

    def h_func_grad(self, a1):
        return (2*self.get_M(a1).dot((a1.state[0:2] - self.state[0:2]))) * self.sign

    def get_state(self):
        return super(Ellipsoid).get_state()
    
    def get_x_dot(self, u=None):
        return self.dyn.get_x_dot(self.state[0:2])

    def get_M(self, obj):
        a_aug = self.a + obj.radius * self.sign
        b_aug = self.b + obj.radius * self.sign
        theta = math.radians(self.angle)

        c = math.cos(theta)
        s = math.sin(theta)

        a = (c/a_aug)**2 + (s/b_aug)**2
        b = -s*c*(1/b_aug**2 - 1/a_aug**2)
        c = (s/a_aug)**2 + (c/b_aug)**2

        return np.array([[a,b],[b,c]])

    def plot(self, ax=None):
        plot_ellipse(self.loc, self.a, self.b, self.angle, ax)
    
    def plot_levelsets(self, ax=None):
        super(Ellipsoid).plot_levelsets(ax=ax)

    def pointFromAngle(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        ta = s / c  ## tan(a)
        a = self.a/2.0
        b = self.b/2.0
        tt = ta * a / b  ## tan(t)
        d = 1. / math.sqrt(1. + tt * tt)
        x = self.loc[0] + math.copysign(a * d, c)
        y = self.loc[1] - math.copysign(b * tt * d, s)
        return x, y

class Wall(Obstacle):
    def __init__(self, x, y, n, d):
        n = make_column(n/np.linalg.norm(n))
        self.n = n
        self.d = d
        self.k_cbf = 1.0

        minx = np.min(x)
        maxx = np.max(x)
        miny = np.min(y)
        maxy = np.max(y)

        # Sign of the line function
        fsign = lambda x,y: np.sign(np.transpose(n).dot(np.array([[x,y]]).T) + d)[0]

        # Find sign of the line function at the 4 corners of the region
        fll=fsign(minx,miny)
        flh=fsign(minx,maxy)
        fhh=fsign(maxx,maxy)
        fhl=fsign(maxx,miny)

        # Make sure line is visible
        if not (fll==flh and flh==fhh and fhh==fhl):
            # Find points of the line at the intersection with the boundaries
            xp = np.zeros(n.shape)
            yp = np.zeros(n.shape)
            p=0 # counts which intersection we are looking for (first or second)
            if fll!=fhl: # south
                yp[p]=miny
                # solve n[0]x+n[1]y+d=0
                xp[p]=(-n[1]*yp[p]-d)/n[0]
                p+=1
            if fll!=flh: # west
                xp[p]=minx
                yp[p]=(-n[0]*xp[p]-d)/n[1]
                p+=1
            if fhl!=fhh: # east
                if p<2:
                    xp[p]=maxx
                    yp[p]=(-n[0]*xp[p]-d)/n[1]
                    p+=1
            if flh!=fhh: # north
                if p<2:
                    yp[p]=maxy
                    xp[p]=(-n[1]*yp[p]-d)/n[0]
                    p+=1
        self.xp = xp
        self.yp = yp

        # TODO: generalize to three dimension
        loc = (np.mean(self.xp), np.mean(self.yp))
        self.dyn = SingleIntegrator(loc)
        self.state = self.dyn.cur_state
        self.h = Function(self.h_func, self.h_func_grad)

    def h_func(self,a1):
        x = a1.state - self.get_state()
        return np.abs(np.transpose(self.n).dot(x)[0][0]) - a1.radius

    def h_func_grad(self, a1):
        return -self.n

    def get_x_dot(self, u=None):
        return self.dyn.get_x_dot(self.state)

    def get_state(self):
        return self.dyn.cur_state

    def plot_levelsets(self, ax=None):
        self.h.plot_levelsets(self.state, ax)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.xp, self.yp, "-r",linewidth=3)
        ax.axis('equal')

def plot_ellipse(loc, x_rad, y_rad=None, angle=0, ax=None):
    if y_rad is None:
        y_rad = x_rad

    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x_rad * math.cos(math.radians(d)) for d in deg]
    yl = [y_rad * math.sin(math.radians(d)) for d in deg]
    xy_arr = np.array([xl, yl]).transpose()

    theta = np.radians(angle)
    if theta % math.pi != 0:
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array(((c, s), (-s, c)))
        xy_arr = xy_arr.dot(rot_mat)

    x = xy_arr[:,0] + loc[0]
    y = xy_arr[:,1] + loc[1]
    
    if ax is None:
        _, ax = plt.subplots()
    
    ax.plot(x, y, "-r",linewidth=3)
    ax.axis('equal')

def make_column(vec):
    vec = np.array(vec)
    vec.shape = (max(vec.shape),1)
    return vec

def main():
    Params('configs.ini')
    d = Dyn.UNICYCLE_EXT
    sphere1 = Sphere(loc=(0,0),dynamics=d)
    a1 = Agent((0,0),(1,1),dynamics=Dyn.UNICYCLE_EXT)

    sphere1.h_func(a1)

    wall1 = Wall((0,5),(0,5), np.array([[1,-1]]), 1)
    wall1.plot(plt.gca())
    plt.pause(3)
    wall1.plot_levelsets(plt.gca())
    plt.pause(3)
    
    # r = .5
    # for i in range(0,1000,15):
    #     r += .1
    #     o = Ellipsoid((0,0), (3*r,r), i)
    #     o.plot(plt.gca())
    #     plt.pause(.1)
    

if __name__ == '__main__':
    main()