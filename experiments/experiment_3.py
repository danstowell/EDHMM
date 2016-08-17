# this experiment sees to see if we can distinguish between two different 
# states with the same emissions but different durations

import pylab as pb
import numpy as np
import logging

import sys
sys.path.append('..')

from emission import Gaussian
from duration import Poisson
from transition import Transition
from initial import Initial
from edhmm import EDHMM


logging.basicConfig(
    stream=sys.stdout,
    filename="experiment_3.log", 
    filemode="w",
    level=logging.DEBUG
)

A = Transition(
    K=3,
    A=pb.array([[0, 0.3, 0.7], [0.6, 0, 0.4], [0.3, 0.7, 0]])
)
O = Gaussian(
    nu = 1,
    Lambda = np.array([1]), 
    mu_0 = [0, 0, 0], 
    kappa = 0.01, 
    mu = [0, 0, 3], 
    tau = [
        np.array([[1]]),
        np.array([[1]]),
        np.array([[1]])
    ]
)
D = Poisson(
    mu = [5,15,20], 
    alpha=[1, 1, 1],
    beta=[1e-5, 1e-5, 1e-5],
    support_step = 20
)

pi = Initial(K=3,beta=0.001)
m = EDHMM(A,O,D,pi)

T = 500

X,Y,Dseq = m.sim(T)

m.A.A = pb.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
m.O.mu = [0,0,0]
m.D.mu = [1,1,1]

np.save("exp3_X.npy", X)
np.save("exp3_D.npy", Dseq)
np.save("exp3_Y.npy", Y)
np.save("exp3_Z.npy", zip(X,Dseq))
L = m.beam(
    [Y], its=3000, burnin=500, name = "exp3", online=True
)
np.save("exp3_L", L)
