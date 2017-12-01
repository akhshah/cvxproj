# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 14:13:29 2017

@author: Karabag
"""

import numpy as np
import math as m
import numpy.linalg as la





def main():
    P = nominal_transitions(10, 5)
    R = np.zeros(100)
    R[0] = 100;
    vals, bestAction = valitr(P, R, 0.5, 1e-3)


def valitr(P, R, discount, eps):
    numOfStates = P.shape[2]
    numOfActions = P.shape[0]
    stateVals = np.zeros(numOfStates)
    bestAction = np.zeros(numOfStates)
    #df is the stopping criteria
    df = np.zeros(1)
    df[0] = 1
    while la.norm(df,2) > eps:
        stateValsNew = np.zeros(P.shape[2])
        #for each state maximize the value
        for s in xrange(numOfStates):
            maxVal = -np.inf;
            bestActOfs = -1;
            #search every action
            for a in xrange(numOfActions):
                successors = np.where(P[a,s,:] > 0)
                valOfa = R[s] + discount*np.dot(stateVals[successors], np.squeeze(P[a,s,successors]))
                if valOfa > maxVal:
                    maxVal = valOfa;
                    bestActOfs = a;
            bestAction[s] = bestActOfs
            stateValsNew[s] = maxVal;
        #update the state values and continue
        df = stateValsNew - stateVals
        stateVals = stateValsNew
    
    return stateVals, bestAction

def nominal_transitions(S,A):
    State=S
    Actions=A
    np.random.seed(0)
    P_0=np.zeros((Actions,State**2,State**2))
    for k in range(P_0.shape[0]):
        for i in range(P_0.shape[1]):
                if i%State==0:
                    if i!=0 and i!=State**2-State:
                        P_0[k,i,i]=np.random.rand(1)
                        P_0[k,i,i+1]=np.random.rand(1)
                        P_0[k,i,i+State]=np.random.rand(1)
                        P_0[k,i,i-State]=np.random.rand(1)
                    if i==0:
                        P_0[k,i,i]=np.random.rand(1)
                        P_0[k,i,i+1]=np.random.rand(1)
                        P_0[k,i,i+State]=np.random.rand(1)
                    if i==State**2-State:
                        P_0[k,i,i]=np.random.rand(1)
                        P_0[k,i,i+1]=np.random.rand(1)
                        P_0[k,i,i-State]=np.random.rand(1)
                elif i%State==State-1:
                    if i!=State-1 and i!=State**2-1:
                        P_0[k,i,i]=np.random.rand(1)
                        P_0[k,i,i-1]=np.random.rand(1)
                        P_0[k,i,i+State]=np.random.rand(1)
                        P_0[k,i,i-State]=np.random.rand(1)
                    if i==State-1:
                        P_0[k,i,i]=np.random.rand(1)
                        P_0[k,i,i-1]=np.random.rand(1)
                        P_0[k,i,i+State]=np.random.rand(1)
                    if i==State**2-1:
                        P_0[k,i,i]=np.random.rand(1)
                        P_0[k,i,i-1]=np.random.rand(1)
                        P_0[k,i,i-State]=np.random.rand(1)
                elif i>0 and i<State:
                    P_0[k,i,i]=np.random.rand(1)
                    P_0[k,i,i-1]=np.random.rand(1)
                    P_0[k,i,i+1]=np.random.rand(1)
                    P_0[k,i,i+State]=np.random.rand(1)
                elif i>(State-1)*State and i<State**2-1:
                    P_0[k,i,i]=np.random.rand(1)
                    P_0[k,i,i-1]=np.random.rand(1)
                    P_0[k,i,i+1]=np.random.rand(1)
                    P_0[k,i,i-State]=np.random.rand(1)
                else:
                    P_0[k,i,i]=np.random.rand(1)
                    P_0[k,i,i-1]=np.random.rand(1)
                    P_0[k,i,i+1]=np.random.rand(1)
                    P_0[k,i,i-State]=np.random.rand(1)
                    P_0[k,i,i+State]=np.random.rand(1)
    for k in range(P_0.shape[0]):
        for i in range(P_0.shape[1]):
            P_0[k,i,:]=P_0[k,i,:]/np.sum(P_0[k,i,:])
    return P_0

if __name__ == "__main__":
    main()