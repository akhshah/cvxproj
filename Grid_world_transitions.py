import numpy as np
import math as m
# This is the nominal transition matrix generator for k x n x n matrix.
# k is the number of actions, i.e. (left,right,up,down,loop)
# Matrices are formed in a way that 0th row of the matrices represent bottom left corner of the grid world
# Similarly last row is for the upper right corner (n^2 th grid)
# The following random matrices allow stochastic transitions for all action types, for example,
# in a state, if the action loop is taken it is possible to go left,right,up,down or loop depending on the position
# of the state in the grid world. In the corners taking some actions would not be possible, 0th grid cant take action down.
# All rows sum up to 1 so it is a well defined transition matrix.
def nominal_transitions(S,A):
    State=S
    Actions=A
    np.random.seed(0)
    prob=0.8
    P_0=np.zeros((Actions,State**2,State**2))
    #action left
    for i in range(P_0.shape[1]):
            if i%State==0:
                if i!=0 and i!=State**2-State:
                    P_0[0,i,i]=prob+(1-prob)/4
                    P_0[0,i,i+1]=(1-prob)/4
                    P_0[0,i,i+State]=(1-prob)/4
                    P_0[0,i,i-State]=(1-prob)/4
                if i==0:
                    P_0[0,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[0,i,i+1]=(1-prob)/4
                    P_0[0,i,i+State]=(1-prob)/4
                if i==State**2-State:
                    P_0[0,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[0,i,i+1]=(1-prob)/4
                    P_0[0,i,i-State]=(1-prob)/4
            elif i%State==State-1:
                if i!=State-1 and i!=State**2-1:
                    P_0[0,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[0,i,i-1]=prob
                    P_0[0,i,i+State]=(1-prob)/4
                    P_0[0,i,i-State]=(1-prob)/4
                if i==State-1:
                    P_0[0,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[0,i,i-1]=prob
                    P_0[0,i,i+State]=(1-prob)/4
                if i==State**2-1:
                    P_0[0,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[0,i,i-1]=prob
                    P_0[0,i,i-State]=(1-prob)/4
            elif i>0 and i<State-1:
                P_0[0,i,i]=(1-prob)/4+(1-prob)/4
                P_0[0,i,i-1]=prob
                P_0[0,i,i+1]=(1-prob)/4
                P_0[0,i,i+State]=(1-prob)/4
            elif i>(State-1)*State and i<State**2-1:
                P_0[0,i,i]=(1-prob)/4+(1-prob)/4
                P_0[0,i,i-1]=prob
                P_0[0,i,i+1]=(1-prob)/4
                P_0[0,i,i-State]=(1-prob)/4
            else:
                P_0[0,i,i]=(1-prob)/4
                P_0[0,i,i-1]=prob
                P_0[0,i,i+1]=(1-prob)/4
                P_0[0,i,i-State]=(1-prob)/4
                P_0[0,i,i+State]=(1-prob)/4
    # action right
    for i in range(P_0.shape[1]):
            if i%State==0:
                if i!=0 and i!=State**2-State:
                    P_0[1,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[1,i,i+1]=prob
                    P_0[1,i,i+State]=(1-prob)/4
                    P_0[1,i,i-State]=(1-prob)/4
                if i==0:
                    P_0[1,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[1,i,i+1]=prob
                    P_0[1,i,i+State]=(1-prob)/4
                if i==State**2-State:
                    P_0[1,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[1,i,i+1]=prob
                    P_0[1,i,i-State]=(1-prob)/4
            elif i%State==State-1:
                if i!=State-1 and i!=State**2-1:
                    P_0[1,i,i]=prob+(1-prob)/4
                    P_0[1,i,i-1]=(1-prob)/4
                    P_0[1,i,i+State]=(1-prob)/4
                    P_0[1,i,i-State]=(1-prob)/4
                if i==State-1:
                    P_0[1,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[1,i,i-1]=(1-prob)/4
                    P_0[1,i,i+State]=(1-prob)/4
                if i==State**2-1:
                    P_0[1,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[1,i,i-1]=(1-prob)/4
                    P_0[1,i,i-State]=(1-prob)/4
            elif i>0 and i<State-1:
                P_0[1,i,i]=(1-prob)/4+(1-prob)/4
                P_0[1,i,i-1]=(1-prob)/4
                P_0[1,i,i+1]=prob
                P_0[1,i,i+State]=(1-prob)/4
            elif i>(State-1)*State and i<State**2-1:
                P_0[1,i,i]=(1-prob)/4+(1-prob)/4
                P_0[1,i,i-1]=(1-prob)/4
                P_0[1,i,i+1]=prob
                P_0[1,i,i-State]=(1-prob)/4
            else:
                P_0[1,i,i]=(1-prob)/4
                P_0[1,i,i-1]=(1-prob)/4
                P_0[1,i,i+1]=prob
                P_0[1,i,i-State]=(1-prob)/4
                P_0[1,i,i+State]=(1-prob)/4
    # action up
    for i in range(P_0.shape[1]):
            if i%State==0:
                if i!=0 and i!=State**2-State:
                    P_0[2,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[2,i,i+1]=(1-prob)/4
                    P_0[2,i,i+State]=prob
                    P_0[2,i,i-State]=(1-prob)/4
                if i==0:
                    P_0[2,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[2,i,i+1]=(1-prob)/4
                    P_0[2,i,i+State]=prob
                if i==State**2-State:
                    P_0[2,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[2,i,i+1]=(1-prob)/4
                    P_0[2,i,i-State]=(1-prob)/4
            elif i%State==State-1:
                if i!=State-1 and i!=State**2-1:
                    P_0[2,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[2,i,i-1]=(1-prob)/4
                    P_0[2,i,i+State]=prob
                    P_0[2,i,i-State]=(1-prob)/4
                if i==State-1:
                    P_0[2,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[2,i,i-1]=(1-prob)/4
                    P_0[2,i,i+State]=prob
                if i==State**2-1:
                    P_0[2,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[2,i,i-1]=(1-prob)/4
                    P_0[2,i,i-State]=(1-prob)/4
            elif i>0 and i<State-1:
                P_0[2,i,i]=(1-prob)/4+(1-prob)/4
                P_0[2,i,i-1]=(1-prob)/4
                P_0[2,i,i+1]=(1-prob)/4
                P_0[2,i,i+State]=prob
            elif i>(State-1)*State and i<State**2-1:
                P_0[2,i,i]=prob+(1-prob)/4
                P_0[2,i,i-1]=(1-prob)/4
                P_0[2,i,i+1]=(1-prob)/4
                P_0[2,i,i-State]=(1-prob)/4
            else:
                P_0[2,i,i]=(1-prob)/4
                P_0[2,i,i-1]=(1-prob)/4
                P_0[2,i,i+1]=(1-prob)/4
                P_0[2,i,i-State]=(1-prob)/4
                P_0[2,i,i+State]=prob
    # action down
    for i in range(P_0.shape[1]):
            if i%State==0:
                if i!=0 and i!=State**2-State:
                    P_0[3,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[3,i,i+1]=(1-prob)/4
                    P_0[3,i,i+State]=(1-prob)/4
                    P_0[3,i,i-State]=prob
                if i==0:
                    P_0[3,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[3,i,i+1]=(1-prob)/4
                    P_0[3,i,i+State]=(1-prob)/4
                if i==State**2-State:
                    P_0[3,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[3,i,i+1]=(1-prob)/4
                    P_0[3,i,i-State]=prob
            elif i%State==State-1:
                if i!=State-1 and i!=State**2-1:
                    P_0[3,i,i]=(1-prob)/4+(1-prob)/4
                    P_0[3,i,i-1]=(1-prob)/4
                    P_0[3,i,i+State]=(1-prob)/4
                    P_0[3,i,i-State]=prob
                if i==State-1:
                    P_0[3,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[3,i,i-1]=(1-prob)/4
                    P_0[3,i,i+State]=(1-prob)/4
                if i==State**2-1:
                    P_0[3,i,i]=(1-prob)/4+(1-prob)/4+(1-prob)/4
                    P_0[3,i,i-1]=(1-prob)/4
                    P_0[3,i,i-State]=prob
            elif i>0 and i<State-1:
                P_0[3,i,i]=prob+(1-prob)/4
                P_0[3,i,i-1]=(1-prob)/4
                P_0[3,i,i+1]=(1-prob)/4
                P_0[3,i,i+State]=(1-prob)/4
            elif i>(State-1)*State and i<State**2-1:
                P_0[3,i,i]=(1-prob)/4+(1-prob)/4
                P_0[3,i,i-1]=(1-prob)/4
                P_0[3,i,i+1]=(1-prob)/4
                P_0[3,i,i-State]=prob
            else:
                P_0[3,i,i]=(1-prob)/4
                P_0[3,i,i-1]=(1-prob)/4
                P_0[3,i,i+1]=(1-prob)/4
                P_0[3,i,i-State]=prob
                P_0[3,i,i+State]=(1-prob)/4
    # action loop
    for i in range(P_0.shape[1]):
            if i%State==0:
                if i!=0 and i!=State**2-State:
                    P_0[4,i,i]=prob+(1-prob)/4
                    P_0[4,i,i+1]=(1-prob)/4
                    P_0[4,i,i+State]=(1-prob)/4
                    P_0[4,i,i-State]=(1-prob)/4
                if i==0:
                    P_0[4,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[4,i,i+1]=(1-prob)/4
                    P_0[4,i,i+State]=(1-prob)/4
                if i==State**2-State:
                    P_0[4,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[4,i,i+1]=(1-prob)/4
                    P_0[4,i,i-State]=(1-prob)/4
            elif i%State==State-1:
                if i!=State-1 and i!=State**2-1:
                    P_0[4,i,i]=prob+(1-prob)/4
                    P_0[4,i,i-1]=(1-prob)/4
                    P_0[4,i,i+State]=(1-prob)/4
                    P_0[4,i,i-State]=(1-prob)/4
                if i==State-1:
                    P_0[4,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[4,i,i-1]=(1-prob)/4
                    P_0[4,i,i+State]=(1-prob)/4
                if i==State**2-1:
                    P_0[4,i,i]=prob+(1-prob)/4+(1-prob)/4
                    P_0[4,i,i-1]=(1-prob)/4
                    P_0[4,i,i-State]=(1-prob)/4
            elif i>0 and i<State-1:
                P_0[4,i,i]=prob+(1-prob)/4
                P_0[4,i,i-1]=(1-prob)/4
                P_0[4,i,i+1]=(1-prob)/4
                P_0[4,i,i+State]=(1-prob)/4
            elif i>(State-1)*State and i<State**2-1:
                P_0[4,i,i]=prob+(1-prob)/4
                P_0[4,i,i-1]=(1-prob)/4
                P_0[4,i,i+1]=(1-prob)/4
                P_0[4,i,i-State]=(1-prob)/4
            else:
                P_0[4,i,i]=prob
                P_0[4,i,i-1]=(1-prob)/4
                P_0[4,i,i+1]=(1-prob)/4
                P_0[4,i,i-State]=(1-prob)/4
                P_0[4,i,i+State]=(1-prob)/4
    return P_0
