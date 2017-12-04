# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 14:13:29 2017

@author: Karabag
"""

import numpy as np
import math as m
import numpy.linalg as la





def main():
    Rows=100
    Columns=100
    numOfStates=Rows*Columns
    #P = nominal_transitions(numOfStates, 5)
    P_0=Grid_world(Rows,Columns,0.8)

    #vals2, bestAction2, error = policyitr(P, R, 0.5, 1e-3)
    
    A=np.zeros((5*numOfStates,numOfStates)) # 5 is number of actions !
    b1=np.zeros([5*numOfStates,1])
    discount = 0.5

    for a in range(5):
        A[a*numOfStates:(a+1)*numOfStates,:]=np.eye(numOfStates)-discount*P_0[a,:,:];
    for a in range(5):
        b1[(a+1)*numOfStates-1,0]=1
    R = b1.copy()
    R = R[0:numOfStates, 0];
    #vals1, bestAction1, error = policyitr(P_0, R, 0.5, 1e-3)
    #print vals1
    vals2, bestAction2 = valitr(P_0, R, 0.5, 1e-3)
    print vals2
    vals3, error = interiorPoint(A, b1, 0.5,1e-3, 0.5, vals2)
    print vals3
    k = 2
    
def plotGridWorld(R, bestAction, Rows, Columns):
    imgData = np.resize(R,[Rows, Columns]);
    imgData = np.flipud(imgData);
    print imgData
    
    plt.imshow(imgData, interpolation='nearest')
    bestAction = np.resize(bestAction[Rows, Columns])
    for a in range(Rows):
        for b in range(Columns): 
            if bestAction[Rows -1 - a,b]== 0:
                plt.text(b,a ,r'$ \leftarrow $')
            elif bestAction[Rows -1 - a,b]== 1:
                plt.text(b,a,r'$ \rightarrow $')
            elif bestAction[Rows -1 - a,b]== 2:
                plt.text(b,a,r'$ \uparrow $')
            elif bestAction[Rows -1 - a,b] == 3:
                plt.text(b,a,r'$ \downarrow $')
            elif bestAction[Rows -1 - a,b] == 4:
                plt.text(b,a ,r'$ o $')
            #plt.text(b,a,r'$ \leftarrow $')
            
    plt.colorbar()
    plt.show()
    return

def findBarrierGradient(A,b,x):
    #calculates the gradient of barrier function for IPM Ax>=b
    d = np.zeros([A.shape[0],1])
    for i in range(A.shape[0]):
        d[i,0] = -1.0/(np.dot(A[i,:],x) - b[i])
    g = np.dot(np.transpose(A),d)
    return g

def findBarrierHessian(A,b,x):
    #calculates the hessian of barrier function for IPM Ax>=b
    d = np.zeros([A.shape[0],1])
    dMat = np.zeros([A.shape[0],A.shape[0]])
    for i in range(A.shape[0]):
        dMat[i,i] = 1.0/((np.dot(A[i,:],x) - b[i])**2)  
    dummy = np.dot(np.transpose(A),dMat)
    H = np.dot(dummy,A)
    return H

def findAnalyticCenter(A,b,xFeasible,T):
    #finds the analytical center for IPM using gradient descent
    for i in range (T):
        g = findBarrierGradient(A,b,xFeasible)
        xFeasible = xFeasible - 0.1/(i+1)*g;
    return xFeasible

def findOptimalSolution(A,b,stateValsNew,alpha,eps, stateValsOpt):
    #finds the optimal solution using IPM
    error = []
    t = 1;
    #to make it enter the loop
    stateVals = stateValsNew + 1;
    itr = 1;
    while (np.linalg.norm(stateVals-stateValsNew,2) > eps):
        stateVals = stateValsNew
        #Finds the gradient and hessian
        g = t*np.ones([stateVals.size, 1]) + findBarrierGradient(A,b,stateVals)
        #H = findBarrierHessian(A,b,stateVals)
        #Updates the calues
        #stateValsNew = stateVals - np.dot(np.linalg.inv(H),g)
        stateValsNew = stateVals - 1/np.sqrt(itr)*g;
        error.append(np.linalg.norm(stateValsNew - stateValsOpt,2))
        t = t*(1+alpha)
        if np.isnan(error[-1]):
                return stateVals , error
    return stateVals, error


def interiorPoint(A, b, discount, eps, alpha, stateValsOpt):
    #finds another feasible point using the optimal solution
    xFeasible = stateValsOpt +1;
    #finds analytical center
    xF = findAnalyticCenter(A,b,xFeasible,1000);
    stateVals, error = findOptimalSolution(A,b,xF,alpha,eps, stateValsOpt)
    return stateVals, error

def policyitr(P, R, discount, eps):
    numOfStates = P.shape[2]
    numOfActions = P.shape[0]
    bestActionOld =  -1*np.ones(numOfStates)
    bestActionNew = np.random.randint(numOfActions, size=numOfStates)
    stateVals = np.zeros([numOfStates, 1]);
    vals = []; #to keep state values;
    #stops if the last two polies are same or
    df = np.zeros(1)
    df[0] = 1
    while not(np.array_equal(bestActionOld, bestActionNew) or la.norm(df,2) < eps):
        bestActionOld =np.copy(bestActionNew);
        #constructs linear equation to find new values of states
        A = np.zeros([numOfStates, numOfStates])
        #if there are rewards for actions this line must be changed
        b = R;
        b.resize([numOfStates,1])
        for s in xrange(numOfStates):
            ts = discount*P[bestActionNew[s],s,:];
            ts.resize([1,numOfStates]);
            A[s,:] = -ts;
        #update the state values and continue
        A = A + np.identity(numOfStates);
        stateValsNew = np.linalg.solve(A,b);
        df = stateValsNew-stateVals;
        stateVals = stateValsNew
        vals.append(stateVals)
        #for each state maximize the value
        for s in xrange(numOfStates):
            maxVal = -np.inf;
            bestActOfs = -1;
            #search every action
            for a in xrange(numOfActions):
                successors = np.where(P[a,s,:] > 0)
                valOfa = R[s] + discount*np.dot(np.squeeze(stateVals[successors]), np.squeeze(P[a,s,successors]))
                if valOfa > maxVal:
                    maxVal = valOfa;
                    bestActOfs = a;
            bestActionNew[s] = bestActOfs
    error = []
    for i in xrange(len(vals)):
        error.append(np.linalg.norm(vals[i] - vals[-1], 2))
    return stateVals, bestActionNew, error
    
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
    stateVals.resize([numOfStates,1])
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

def Grid_world(Row,Col,Prob):
    State=Row*Col
    Actions=5
    np.random.seed(0)
    prob=Prob
    P_0=np.zeros((Actions,State,State))
    #action left
    for i in range(P_0.shape[1]):
            if i%Col==0:
                if i!=0 and i!=State-Col:
                    P_0[0,i,i]=prob+(1-prob)/4
                    P_0[0,i,i+1]=(1-prob)/4
                    P_0[0,i,i+Col]=(1-prob)/4
                    P_0[0,i,i-Col]=(1-prob)/4
                if i==0:
                    P_0[0,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[0,i,i+1]=(1-prob)/4
                    P_0[0,i,i+Col]=(1-prob)/4
                if i==State-Col:
                    P_0[0,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[0,i,i+1]=(1-prob)/4
                    P_0[0,i,i-Col]=(1-prob)/4
            elif i%Col==Col-1:
                if i!=Col-1 and i!=State-1:
                    P_0[0,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[0,i,i-1]=prob
                    P_0[0,i,i+Col]=(1-prob)/4
                    P_0[0,i,i-Col]=(1-prob)/4
                if i==Col-1:
                    P_0[0,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[0,i,i-1]=prob
                    P_0[0,i,i+Col]=(1-prob)/4
                if i==State-1:
                    P_0[0,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[0,i,i-1]=prob
                    P_0[0,i,i-Col]=(1-prob)/4
            elif i>0 and i<Col-1:
                P_0[0,i,i]=(1-prob)/4+(1-prob)/4
                P_0[0,i,i-1]=prob
                P_0[0,i,i+1]=(1-prob)/4
                P_0[0,i,i+Col]=(1-prob)/4
            elif i>(Row-1)*Col and i<State-1:
                P_0[0,i,i]=(1-prob)/4+(1-prob)/4
                P_0[0,i,i-1]=prob
                P_0[0,i,i+1]=(1-prob)/4
                P_0[0,i,i-Col]=(1-prob)/4
            else:
                P_0[0,i,i]=(1-prob)/4
                P_0[0,i,i-1]=prob
                P_0[0,i,i+1]=(1-prob)/4
                P_0[0,i,i-Col]=(1-prob)/4
                P_0[0,i,i+Col]=(1-prob)/4
    # action right
    for i in range(P_0.shape[1]):
            if i%Col==0:
                if i!=0 and i!=State-Col:
                    P_0[1,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[1,i,i+1]=prob
                    P_0[1,i,i+Col]=(1-prob)/4
                    P_0[1,i,i-Col]=(1-prob)/4
                if i==0:
                    P_0[1,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[1,i,i+1]=prob
                    P_0[1,i,i+Col]=(1-prob)/4
                if i==State-Col:
                    P_0[1,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[1,i,i+1]=prob
                    P_0[1,i,i-Col]=(1-prob)/4
            elif i%Col==Col-1:
                if i!=Col-1 and i!=State-1:
                    P_0[1,i,i]=prob+(1-prob)/4
                    P_0[1,i,i-1]=(1-prob)/4
                    P_0[1,i,i+Col]=(1-prob)/4
                    P_0[1,i,i-Col]=(1-prob)/4
                if i==Col-1:
                    P_0[1,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[1,i,i-1]=(1-prob)/4
                    P_0[1,i,i+Col]=(1-prob)/4
                if i==State-1:
                    P_0[1,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[1,i,i-1]=(1-prob)/4
                    P_0[1,i,i-Col]=(1-prob)/4
            elif i>0 and i<Col-1:
                P_0[1,i,i]=(1-prob)/4+(1-prob)/4
                P_0[1,i,i-1]=(1-prob)/4
                P_0[1,i,i+1]=prob
                P_0[1,i,i+Col]=(1-prob)/4
            elif i>(Row-1)*Col and i<State-1:
                P_0[1,i,i]=(1-prob)/4+(1-prob)/4
                P_0[1,i,i-1]=(1-prob)/4
                P_0[1,i,i+1]=prob
                P_0[1,i,i-Col]=(1-prob)/4
            else:
                P_0[1,i,i]=(1-prob)/4
                P_0[1,i,i-1]=(1-prob)/4
                P_0[1,i,i+1]=prob
                P_0[1,i,i-Col]=(1-prob)/4
                P_0[1,i,i+Col]=(1-prob)/4
    # action up
    for i in range(P_0.shape[1]):
            if i%Col==0:
                if i!=0 and i!=State-Col:
                    P_0[2,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[2,i,i+1]=(1-prob)/4
                    P_0[2,i,i+Col]=prob
                    P_0[2,i,i-Col]=(1-prob)/4
                if i==0:
                    P_0[2,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[2,i,i+1]=(1-prob)/4
                    P_0[2,i,i+Col]=prob
                if i==State-Col:
                    P_0[2,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[2,i,i+1]=(1-prob)/4
                    P_0[2,i,i-Col]=(1-prob)/4
            elif i%Col==Col-1:
                if i!=Col-1 and i!=State-1:
                    P_0[2,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[2,i,i-1]=(1-prob)/4
                    P_0[2,i,i+Col]=prob
                    P_0[2,i,i-Col]=(1-prob)/4
                if i==Col-1:
                    P_0[2,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[2,i,i-1]=(1-prob)/4
                    P_0[2,i,i+Col]=prob
                if i==State-1:
                    P_0[2,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[2,i,i-1]=(1-prob)/4
                    P_0[2,i,i-Col]=(1-prob)/4
            elif i>0 and i<Col-1:
                P_0[2,i,i]=(1-prob)/4+(1-prob)/4
                P_0[2,i,i-1]=(1-prob)/4
                P_0[2,i,i+1]=(1-prob)/4
                P_0[2,i,i+Col]=prob
            elif i>(Row-1)*Col and i<State-1:
                P_0[2,i,i]=prob+(1-prob)/4
                P_0[2,i,i-1]=(1-prob)/4
                P_0[2,i,i+1]=(1-prob)/4
                P_0[2,i,i-Col]=(1-prob)/4
            else:
                P_0[2,i,i]=(1-prob)/4
                P_0[2,i,i-1]=(1-prob)/4
                P_0[2,i,i+1]=(1-prob)/4
                P_0[2,i,i-Col]=(1-prob)/4
                P_0[2,i,i+Col]=prob
    # action down
    for i in range(P_0.shape[1]):
            if i%Col==0:
                if i!=0 and i!=State-Col:
                    P_0[3,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[3,i,i+1]=(1-prob)/4
                    P_0[3,i,i+Col]=(1-prob)/4
                    P_0[3,i,i-Col]=prob
                if i==0:
                    P_0[3,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[3,i,i+1]=(1-prob)/4
                    P_0[3,i,i+Col]=(1-prob)/4
                if i==State-Col:
                    P_0[3,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[3,i,i+1]=(1-prob)/4
                    P_0[3,i,i-Col]=prob
            elif i%Col==Col-1:
                if i!=Col-1 and i!=State-1:
                    P_0[3,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[3,i,i-1]=(1-prob)/4
                    P_0[3,i,i+Col]=(1-prob)/4
                    P_0[3,i,i-Col]=prob
                if i==Col-1:
                    P_0[3,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[3,i,i-1]=(1-prob)/4
                    P_0[3,i,i+Col]=(1-prob)/4
                if i==State-1:
                    P_0[3,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[3,i,i-1]=(1-prob)/4
                    P_0[3,i,i-Col]=prob
            elif i>0 and i<Col-1:
                P_0[3,i,i]=prob+(1-prob)/4
                P_0[3,i,i-1]=(1-prob)/4
                P_0[3,i,i+1]=(1-prob)/4
                P_0[3,i,i+Col]=(1-prob)/4
            elif i>(Row-1)*Col and i<State-1:
                P_0[3,i,i]=(1-prob)/4+(1-prob)/4
                P_0[3,i,i-1]=(1-prob)/4
                P_0[3,i,i+1]=(1-prob)/4
                P_0[3,i,i-Col]=prob
            else:
                P_0[3,i,i]=(1-prob)/4
                P_0[3,i,i-1]=(1-prob)/4
                P_0[3,i,i+1]=(1-prob)/4
                P_0[3,i,i-Col]=prob
                P_0[3,i,i+Col]=(1-prob)/4
    # action loop
    for i in range(P_0.shape[1]):
            if i%Col==0:
                if i!=0 and i!=State-Col:
                    P_0[4,i,i]=prob+(1-prob)/4
                    P_0[4,i,i+1]=(1-prob)/4
                    P_0[4,i,i+Col]=(1-prob)/4
                    P_0[4,i,i-Col]=(1-prob)/4
                if i==0:
                    P_0[4,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[4,i,i+1]=(1-prob)/4
                    P_0[4,i,i+Col]=(1-prob)/4
                if i==State-Col:
                    P_0[4,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[4,i,i+1]=(1-prob)/4
                    P_0[4,i,i-Col]=(1-prob)/4
            elif i%Col==Col-1:
                if i!=Col-1 and i!=State-1:
                    P_0[4,i,i]=prob+(1-prob)/4
                    P_0[4,i,i-1]=(1-prob)/4
                    P_0[4,i,i+Col]=(1-prob)/4
                    P_0[4,i,i-Col]=(1-prob)/4
                if i==Col-1:
                    P_0[4,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[4,i,i-1]=(1-prob)/4
                    P_0[4,i,i+Col]=(1-prob)/4
                if i==State-1:
                    P_0[4,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[4,i,i-1]=(1-prob)/4
                    P_0[4,i,i-Col]=(1-prob)/4
            elif i>0 and i<Col-1:
                P_0[4,i,i]=prob+(1-prob)/4
                P_0[4,i,i-1]=(1-prob)/4
                P_0[4,i,i+1]=(1-prob)/4
                P_0[4,i,i+Col]=(1-prob)/4
            elif i>(Row-1)*Col and i<State-1:
                P_0[4,i,i]=prob+(1-prob)/4
                P_0[4,i,i-1]=(1-prob)/4
                P_0[4,i,i+1]=(1-prob)/4
                P_0[4,i,i-Col]=(1-prob)/4
            else:
                P_0[4,i,i]=prob
                P_0[4,i,i-1]=(1-prob)/4
                P_0[4,i,i+1]=(1-prob)/4
                P_0[4,i,i-Col]=(1-prob)/4
                P_0[4,i,i+Col]=(1-prob)/4
    return P_0


if __name__ == "__main__":
    main()
