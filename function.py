import numpy as np
import matplotlib.pyplot as plt

class Function(object):
    def __init__(self, func, grad_func, hess_func=None):
        self.func = func
        self.grad_func = grad_func
        self.hess_func = hess_func

    def eval(self, x):
        #e.g. return x.dot(x)/2
            return self.func(x)

    def grad(self, x):
        #e.g. return x
        return self.grad_func(x)

    def hess(self, x):
        return self.hess_func(x)

    def plot_levelsets(self, center, ax=None):
        class test:
            def __init__(self, state):
                test.state = state
                test.radius = 1
        

        delta = .25

        x = np.arange(-10 + center[0], 10 + center[0], delta)
        y = np.arange(-10 + center[1], 10 + center[1], delta)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        for i in range(len(x)):
            for j in range(len(y)):
                state = np.array([[X[i,j]],[Y[i,j]]])
                t = test(state)
                Z[i,j] = self.eval(t)

        if ax is None:
            fig, ax = plt.subplots()
        
        ax.contourf(X,Y,Z,10)
        plt.pause(5)


    def grad_test(self):
        #define line (curve) along which to evaluate derivative
        x0=np.random.standard_normal(2)
        v0=np.random.standard_normal(2)
        
        x0 = np.array([ [x0[0]], [v0[0]], [x0[1]], [v0[1]] ])
        v0 = np.array([ [v0[0]], [0],     [v0[1]], [0]     ])        
    
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
            func_val[idx] = self.eval(x_eval)
            der_eval[idx] = self.grad(x_eval).transpose().dot(v0)
            der_expected[idx]=(self.eval(x(t_eval[idx]+delta))-self.eval(x_eval))/delta
            

        #plot der_eval, der_expected vs t_eval
        plt.clf()
        plt.subplot()
        plt.plot(t_eval, func_val)
        plt.plot(t_eval, der_eval, linewidth=2)
        plt.plot(t_eval, der_expected, '--', linewidth=2)
        plt.legend(('func', 'der_eval', 'der_expected'))
        plt.show()

def main():
    v_func = lambda x: x.transpose().dot(x)/2
    del_v = lambda x: x
    v = Function(v_func, del_v)

    x = np.array([[1],[2],[3],[4]])

    x = np.array([ 0.81109883, -0.29460953,  0.10796902,  1.16032042])
    x.shape = (4,1)
    print(v.eval(x))
    v_grad = v.grad(x)
    print(v_grad)

    v.grad_test()

    

if __name__ == '__main__':
    main()
    