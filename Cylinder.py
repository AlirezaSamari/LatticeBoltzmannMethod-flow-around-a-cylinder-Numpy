"""
Created on Mon Aug  1 02:23:22 2022

@author: Alireza
Source Code: Coursera: The Simulation of Natural processes
"""
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

### Flow Definition ##########################################################
maxIter = 200000 # total number of time iteration
Re = 222        #reynolds number
nx, ny = 420, 180   # number of lattice nodes
ly = ny - 1     # hieght of domain in lattice units
cx, cy, r = nx // 4, ny // 2, ny // 9   # coordinates of the cylinder
uLB = 0.04  # velocity in lattice unites
nulb = (uLB*r)/Re # viscoscity in lattice unites
omega = 1/(3*nulb + 0.5)    #relaxation parameter

### Lattice Constants ########################################################
v = array([[1,1], [1,0], [1,-1],
                                [0,1], [0,0], [0,-1], 
                                                     [-1,1], [-1,0], [-1,-1]])

t = array([[1/36], [1/9], [1/36],
                                 [1/9], [4/9], [1/9],
                                                     [1/36], [1/9], [1/36]])

col1 = array([0,1,2])
col2 = array([3,4,5])
col3 = array([6,7,8])

### Function Definitions #####################################################
def macroscopic(fin):
    rho = sum(fin, axis = 0)
    u = zeros((2, nx, ny))
    for i in range(9):
        u[0, :, :] += v[i, 0] * fin[i, :, :]
        u[1, :, :] += v[i, 1] * fin[i, :, :]
    u /= rho
    return rho, u


def equilibrium(rho, u):
    usqr = (3/2) * (pow(u[0], 2) + pow(u[1],2))
    feq = zeros((9, nx, ny))
    for i in range(9):
        uv = 3* (v[i, 0] * u[0,: ,:] + v[i, 1] * u[1, :, :])
        feq[i, :, :] = rho * t[i] *(1 + uv + 0.5* uv**2 - usqr)
    return feq


def obstacle_fun(x, y):
    return (pow(x - cx, 2) + pow(y - cy, 2) < pow(r, 2))

obstacle = fromfunction(obstacle_fun, (nx, ny))
# initial velocity:
def inivel(d, x, y):
    return (1-d) * uLB * (1 + 1e-4* sin(y/ly * 2 * pi))

    
vel = fromfunction(inivel , (2, nx, ny))

fin = equilibrium(1, vel)


### Main Time Loop ###########################################################
myimages = []
for time in range(maxIter):
    # right wall: outflow condition
    fin[col3, -1, :] = fin[col3, -2, :]
    #compute macroscopic variables, rho and u
    rho, u = macroscopic(fin)
    #left wall inflow condition
    u[: , 0, :] = vel[:, 0, :]
    rho[0, :] = (1/(1-u[0, 0, :])) * (sum(fin[col2, 0, :], axis = 0) + 2 * sum(fin[col3, 0, :], axis = 0))
    # compute equibliruim
    feq = equilibrium(rho, u)
    fin[[0, 1, 2],0 , :] = feq[[0, 1, 2], 0, :] + fin[[8, 7, 6], 0, :] - feq[[8, 7, 6], 0, :]
    # collision step
    fout = fin - omega * (fin - feq)
    # bounce-back condition for obstacle
    for i in range(9):
        fout[i, obstacle] = fin[8 - i , obstacle]
    # streaming step
    
    for i in range(9):
        fin[i, :, :] = roll(roll(fout[i,:,:], v[i,0], axis = 0), v[i, 1], axis = 1)
                

    # visualization of velocity
    if time % 100 == 0:
        plt.clf()
        imgplot = plt.imshow(sqrt(pow(u[0], 2)+pow(u[1], 2)).transpose() , cmap=cm.jet)
        plt.savefig("vel.{0:04d}.png".format(time//25))