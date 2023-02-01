import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import scipy.optimize as so



#______________________________________________________
#gaussian function x = vector of space domain  mu = mean value s= standard deviation
def gauss(x,mu,s):   
        f=norm.pdf(x,mu,s)
        return f
#calculates integral (area under a curve)
def integral(Uin,dx):
     return np.sum(Uin*dx) 
    
def RMSE(exact_solution, num_solution): #Root Mean Square Error

    mean_of_differences_squared = ((exact_solution- num_solution)**2).mean() 

    rmse_val = np.sqrt(mean_of_differences_squared)         

    return rmse_val      



#__________________________________________________________________________________
               # FINITE DIFFERENT SCHEMES 
#__________________________________________________________________________________________


def analytical_solution(dx,limx,dt,limt,K,plot=0):
    #PARAMETERS
#________________________________________________________
#space sampling, finer for the analytical solution (X)
    dX=0.1
    XX=np.arange(0,limx,dX)
    xx=np.arange(0,limx,dx)#space domain for numerical models 

    # time domain  (hours)
    tt=np.arange(0,limt,dt)

#variables 
    s=5  #standard dev   gaussian
    mu=50   #gaussian mean
#________________________________________
    u=np.zeros((len(tt),len(XX)))
    ubis=np.zeros((len(tt),len(xx)))
    
    m = 0.
    listXX=[]
    u[0,:] = gauss(XX, mu, s)
    ubis[0,:] = gauss(xx, mu, s)
    
    U0=gauss(XX, mu, s)
    for it in range(len(tt)):
        t = tt[it]
        ixx=0
        for ix in range(0,len(XX)):
            X=XX[ix]
            m, sig = X, np.sqrt(2*K*t)
            zz = gauss(XX, m, sig)
            u[it,ix] = integral(U0*zz,dX)
            
            if X in xx :
                listXX.append(ix)
                ubis[it,ixx]=u[it,ix]
                ixx+=1    
        if plot==1:
            
            if it%10==0 :
                
                plt.plot(xx,ubis[it,:],label='t='+str(it)+' s')
                plt.legend()
                plt.xlabel('Space domain (cm)')
                plt.ylabel('Température')
                plt.grid()
                plt.title('Heat diffusion')
    return ubis







#euler forward scheme 
def euler_f(dx,limx,dt,limt,K,plot=0):

#_________________,_______________________________________
    #space domain (cm)
    xx=np.arange(0,limx,dx)

    # time domain  (hours)
    tt=np.arange(0,limt,dt)

#variables 
    s=5  #standard dev   gaussian
    mu=50   #gaussian mean
#_________________________________

    Tf=np.zeros((len(tt),len(xx)))
    Tf[0,:]=gauss(xx, mu, s) #Initial condition (temperature in depth) 
    
    Tf[:,0]=np.zeros(len(tt))  # Boundary condition surface
    Tf[:,-1]=np.zeros(len(tt)) #Boundary condition bottom

    for i in range(0,len(tt)-1):
        for j in range(1,len(xx)-1):
            Tf[i+1,j]=Tf[i,j]+(K*dt/(dx**2))*(Tf[i,j+1]+Tf[i,j-1]-2*Tf[i,j])
        
        if plot==1:
            if i%10==0:
                plt.plot(xx,Tf[i,:],label='t='+str(i)+' h')
                plt.legend()
                plt.xlabel('Space domain (cm)')
                plt.ylabel('Temperature (°C)')
                plt.grid()
                plt.title('Heat diffusion')
    return Tf
    

    
    
    
    
#Euler backward scheme (implicit)    
def euler_b(dx,limx,dt,limt,K,plot=0):
#_________________,_______________________________________     
   #space domain (cm)
    xx=np.arange(0,limx,dx)

    # time domain  (hours)
    tt=np.arange(0,limt,dt)

#variables 
    s=5  #standard dev   gaussian
    mu=50   #gaussian mean
#_________________,_______________________________________    
    Tb=np.zeros((len(tt),len(xx)))
#Initiales conditions temperature T(0,x)   gaussian (xo,s)
    Tb[0,:]=gauss(xx, mu, s)
    # interior points
    A=np.zeros((len(xx),len(xx)))

    #first and last points (diagonale)
    A[0,0]=1
    A[len(xx)-1,len(xx)-1]=1
    
#constructions of arrays
    for j in range(1,len(xx)-1):
        A[j,j-1]=-(K*dt)/(dx**2)
        A[j,j]=1+(2*K*dt)/(dx**2)
        A[j,j+1]=-(K*dt)/(dx**2)
    
    RHS=np.zeros(len(xx))
    for i in range(1,len(tt)):
        RHS=Tb[i-1,:]
        RHS[0]=0
        RHS[len(xx)-1]=0
    
     #solving linear system
        Tb[i,:]=np.linalg.solve(A,RHS)
    
        if plot==1:
            if i%10==0:
                plt.plot(xx,Tb[i,:],label='t='+str(i)+' h')
                plt.legend()
                plt.xlabel('Space domain (cm)')
                plt.ylabel('Temperature (°C)')
                plt.grid()
                plt.title('Heat diffusion')
                
    return Tb



def Dufort_Frankel(dx,limx,dt,limt,Tb,K,plot=0):
#_________________,_______________________________________     
   #space domain (cm)
    xx=np.arange(0,limx,dx)

    # time domain  (hours)
    tt=np.arange(0,limt,dt)

#variables 
    s=5  #standard dev   gaussian
    mu=50   #gaussian mean
#_________________,_______________________________________    

    Td=np.zeros((len(tt),len(xx)))
    Td[0,:]=gauss(xx, mu, s) #Initial conditions
    
    #Boundary Conditions  
    Td[:,0]=np.zeros(len(tt))
    Td[:,-1]=np.zeros(len(tt))
    
    A=(K*2*dt)/(dx**2)
    Td[1,:]=Tb[1,:]   #euler background value 
   
    for i in range(2,len(tt)):
        for j in range(0,len(xx)-1):
            Td[i,j]=Td[i-2,j]*((1-A)/(1+A)) + (A/(1+A))* (Td[i-1,j-1]+Td[i-1,j+1])
            
        if plot==1:
            if i%10==0: #plot 1 out of ten
                plt.plot(xx,Td[i,:],label='t='+str(i)+' h')
                plt.legend()
                plt.xlabel('Space domain (cm)')
                plt.ylabel('Temperature')
                plt.grid()
                plt.title('Heat diffusion')
    return Td





#Crank Nicolson scheme, theta scheme =0.5, implicit
# resolution of a linear system with the help of a cost function

#_______________________________________________________________
def LeftOp(X,dx,dt,K): #left part of equation
    theta =0.5
    A = K*dt*theta/(dx**2)
    thA = theta*A
    Z = -thA*np.roll(X,1) -thA*np.roll(X,-1) + (1+2*thA)*X
    return Z

def RightOp(X,dx,dt,K):#right part of equation 
    theta =0.5
    A = K*dt*theta/(dx**2)
    thA = theta*A
    Z = (A-thA)*np.roll(X,1) + (A-thA)*np.roll(X,-1) + (1-2*A+2*thA)*X
    return Z

def CostFunc_CN(X, b,dx,dt,K): #cost function
    return 0.5*np.dot(X,LeftOp(X,dx,dt,K)) - np.dot(X,b)

def CostGrad_CN(X, b,dx,dt,K): # cost gradient 
    return LeftOp(X,dx,dt,K) - b



def Crank_Nicolson(dx,limx,dt,limt,K,plot=0):
#_________________,_______________________________________     
   #space domain (cm)
    xx=np.arange(0,limx,dx)

    # time domain  (hours)
    tt=np.arange(0,limt,dt)

#variables 
    s=5  #standard dev   gaussian
    mu=50   #gaussian mean    
#____________________,_______________________________________    
    Tc=np.zeros((len(tt),len(xx)))
    Tc[0,:]=gauss(xx, mu, s) #Initial conditions
    #Boundary Conditions 
    Tc[:,0]=np.zeros(len(tt))
    Tc[:,-1]=np.zeros(len(tt))
    
    theta =0.5
    A = K*dt*theta/(dx**2)
    thA = theta*A

    for i in range(1,len(tt)):
        b = RightOp(Tc[i-1,:],dx,dt,K)
        Xopt = Tc[i-1,:]
#minimization of the Cost function to find the best fit     
        res = so.minimize(CostFunc_CN, Xopt, args=(b,dx,dt,K),
                          method='L-BFGS-B',jac=CostGrad_CN,
                          options={'maxiter': 20})
        Tc[i,:] = res['x']
        
        if plot==1:
            if i%10==0:
                plt.plot(xx,Tc[i,:],label='t='+str(i)+' h')
                plt.legend()
                plt.xlabel('Space domain (cm)')
                plt.ylabel('Temperature  (°C)')
                plt.grid()
                plt.title('Heat diffusion')
    return Tc





