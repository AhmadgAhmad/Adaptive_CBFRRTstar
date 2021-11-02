import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.neighbors import KernelDensity
import timeit
import random
from sklearn import mixture


def main():
    # generating synthetic datapoints:
    # The covariance matrix:

    #Generate the samples:
    X = []
    theta = np.pi / 4
    u1_T = [np.cos(theta), np.sin(theta)]
    u2_T = [np.sin(theta), -np.cos(theta)]
    U_T = np.array([u1_T, u2_T])
    U = U_T.transpose()
    sigma1 = 30
    sigma2 = 10
    Lambda = np.array([[sigma1, 0], [0, sigma2]])
    cov = U.dot(Lambda.dot(U.transpose()))
    mean = np.array([6, 6])
    for iSample in range(100):
        x, y = np.random.multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)
        # x = random.uniform(-20, 20)
        # y = random.uniform(-20, 20)
        X.append([x,y])




    #The generated points that will be used to estimate the distribution:
    X = np.array(X)
    Xx = X[:,0]
    Xy = X[:,1]
    #Fit the Kernel-density using the sampled points:
    t0 = timeit.default_timer()
    kde = KernelDensity(kernel='gaussian', bandwidth=1.5)
    kde.fit(X) #Now using this estimated KDE evaluate the score for a prespecified grid:
    t1 = timeit.default_timer()
    TM = t1-t0

    #Fit using GMM:
    t0 = timeit.default_timer()
    GMM = mixture.GaussianMixture(n_components=4, covariance_type='full')
    GMM.fit(X)
    t1 = timeit.default_timer()
    TM = t1 - t0
    #Specify the a grid where you expect most of the points that have been used in the estimation are dwelling
    x_gridv = np.linspace(-25,25,100)
    y_gridv = np.linspace(-25,25,100)
    Xxgrid,Xygrid = np.meshgrid(x_gridv,y_gridv)
    XYgrid_mtx = np.array([Xxgrid.ravel(),Xygrid.ravel()]).T
    grid_probs = np.exp(kde.score_samples(XYgrid_mtx))
    grid_probsGMM = np.exp(GMM.score_samples(XYgrid_mtx))

    kdeplot=plt.figure(1)
    CS = plt.contour(Xxgrid, Xygrid, grid_probs.reshape(Xxgrid.shape)) # , norm=LogNorm(vmin=4.18, vmax=267.1))
    plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X[:, 0], X[:, 1])
    kdeplot.show()

    GMMplot = plt.figure(2)
    CS = plt.contour(Xxgrid, Xygrid, grid_probsGMM.reshape(Xxgrid.shape))  # , norm=LogNorm(vmin=4.18, vmax=267.1))
    plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X[:, 0], X[:, 1])
    GMMplot.show()

    pass

if __name__ == '__main__':
    main()