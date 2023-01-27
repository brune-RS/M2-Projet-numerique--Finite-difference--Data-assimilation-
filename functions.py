import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

#______________________________________________________
def gauss(x,mu,s):
        f=norm.pdf(x,mu,s)
        return f
def integral(Uin,dx):
     return np.sum(Uin*dx) 

#___________________________________________
#surface temperature boundary condition
def temp_cycle (tt,dt):

    omega = 2*np.pi/(24) #minutes
    temp = 5 * ( 1.5 + np.sin(omega*tt) )

    m_series = np.empty(len(tt))  # empty array of Nstep values. We could have made an array of zeros with np.zeros, too.
    m = 0.

    for i in range(len(tt)):
        m = m + (temp[i] - 0.5 * m)*dt
        m_series[i] = m
    return m_series

#__________________________________________________________________________________
               # FINITE DIFFERENT SCHEMES 
#__________________________________________________________________________________________
def analytical_solution(dx,limx,dt,limt,K,plot=0):
    #PARAMETERS
#________________________________________________________
    dX=0.1
    #limx=200   #cm
    XX=np.arange(0,limx,dX)
    xx=np.arange(0,limx,dx)

    #domaine temporel  (en h)
    tt=np.arange(0,limt,dt)

#variables 
    s=5  #standard dev   gaussian
    mu=50   #moyenne gaussienne 
    #K=0.1
#________________________________________
    u=np.zeros((len(tt),len(XX)))
    ubis=np.zeros((len(tt),len(xx)))
    
    omega = 2*np.pi/24
    temp = 5 * ( 1.2 + np.sin(omega*tt) )
    m_series = np.empty(len(tt))  # empty array of Nstep values. We could have made an array of zeros with np.zeros, too.
    m = 0.
    listXX=[]
    u[0,:] = gauss(XX, mu, s)
    ubis[0,:] = gauss(xx, mu, s)
    
    U0=gauss(XX, mu, s)
    for it in range(len(tt)):
        t = tt[it]
        ubis[it,0]=temp[it]
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
            if it%20==0:
                plt.plot(xx,ubis[it,:],label='t='+str(it)+' s')
                plt.legend()
                plt.xlabel('Space domain (cm)')
                plt.ylabel('Température')
                plt.grid()
                plt.title('Heat diffusion')
    return ubis



#euler forward function 
def euler_f(dx,limx,dt,limt,K,plot=0):
    #PARAMETERS
#_________________,_______________________________________
#dx=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,10]
    dX=0.1
    #limx=100
    XX=np.arange(0,limx,dX)
    xx=np.arange(0,limx,dx)

    #domaine temporel  (en h)
    tt=np.arange(0,limt,dt)

#variables 
    s=5  #standard dev   gaussian
    mu=50   #moyenne gaussienne 
    #K=0.1
#_________________________________

    Tf=np.zeros((len(tt),len(xx)))
    Tf[0,:]=gauss(xx, mu, s)
    #Conditions aux limites 
    Tf[:,0]=np.zeros(len(tt))
    Tf[:,-1]=np.zeros(len(tt))

    for i in range(0,len(tt)-1):
        for j in range(1,len(xx)-1):
            Tf[i+1,j]=Tf[i,j]+(K*dt/(dx**2))*(Tf[i,j+1]+Tf[i,j-1]-2*Tf[i,j])
        if plot==1:
            if i%10==0:
                plt.plot(xx,Tf[i,:],label='t='+str(i)+' h')
                plt.legend()
                plt.xlabel('Space domain (cm)')
                plt.ylabel('Température')
                plt.grid()
                plt.title('Heat diffusion')
    return Tf
    


    
    
def euler_b(dx,limx,dt,limt,K,plot=0):
     
    xx=np.arange(0,limx,dx)
    #domaine temporel  (en h)
    tt=np.arange(0,limt,dt)

    Tb=np.zeros((len(tt),len(xx)))
    
    s=5  #standard dev   gaussian
    mu=50   #moyenne gaussienne 
    #K=0.1
    

#Conditions initiales température T(0,x)   gaussienne (xo,s)
    Tb[0,:]=gauss(xx, mu, s)
    #points intérieurs
    A=np.zeros((len(xx),len(xx)))

    #premier et derniers points dans la diagonale
    A[0,0]=1
    A[len(xx)-1,len(xx)-1]=1

    for j in range(1,len(xx)-1):
        A[j,j-1]=-(K*dt)/(dx**2)
        A[j,j]=1+(2*K*dt)/(dx**2)
        A[j,j+1]=-(K*dt)/(dx**2)
    
    RHS=np.zeros(len(xx))
    for i in range(1,len(tt)):
        RHS=Tb[i-1,:]
        RHS[0]=0
        RHS[len(xx)-1]=0
    
     #solving system
        Tb[i,:]=np.linalg.solve(A,RHS)
    
        if plot==1:
            if i%10==0:
                plt.plot(xx,Tb[i,:],label='t='+str(i)+' h')
                plt.legend()
                plt.xlabel('Space domain (cm)')
                plt.ylabel('Température')
                plt.grid()
                plt.title('Heat diffusion')
                
    return Tb



def Dufort_Frankel(dx,limx,dt,limt,K):
    xx=np.arange(0,limx,dx)
   
    tt=np.arange(0,limt,dt)

    s=5  #standard dev   gaussian
    mu=50   #moyenne gaussienne 
    #K=0.1

    Td=np.zeros((len(tt),len(xx)))
    Td[0,:]=gauss(xx, mu, s)
    #Td[1,:]=gauss(xx, mu, s)
    #Conditions aux limites 
    Td[:,0]=np.zeros(len(tt))
    #Td[:,1]=np.zeros(len(tt))
    Td[:,-1]=np.zeros(len(tt))
    A=(K*2*dt)/(dx**2)
    Td[1,:]=Tf[1,:]   #euler forward value 
    #Td[1,:] = Td[0,:] + A * (np.roll(Td,-1,axis=1) +np.roll(Td,1,axis=1)-2*np.roll(Td,0,axis=1))[0]
    for i in range(2,len(tt)):

        for j in range(0,len(xx)-1):
            Td[i,j]=Td[i-2,j]*((1-A)/(1+A)) + (A/(1+A))* (Td[i-1,j-1]+Td[i-1,j+1])
        if i%10==0:
            plt.plot(xx,Td[i,:],label='t='+str(i)+' h')
            plt.legend()
            plt.xlabel('Space domain (cm)')
            plt.ylabel('Température')
            plt.grid()
            plt.title('Heat diffusion')
    return Td



#________________________________________________________________________________________
#______________________________________________________________________________________
#create observation files at 5 cm and 10 cm depth 

def obs_5 (dt,limt,freq):  #freq en minutes  
    tt=np.arange(0,limt,dt)
    omega = 2*math.pi/(24) # faire varier la période (stochastique)
    temp = 0.5* ( 12+ np.sin(omega*tt) ) 
    f=freq # en heures pas de temps entre 2 mesures 
    tt_obs=np.arange(0,limt,dt*f)
    tt_obs=tt_obs+10
    m_series = np.empty(len(tt))  # empty array of Nstep values.
    obs_series=np.empty(len(tt_obs))
    
    noise = np.random.normal(0, 0.2, m_series.shape) #bruit du signal pour rajouter erreur de mesure 
    m = 0.

    for i in range(len(tt)):
        m = m + (temp[i] - 0.5 * m)*dt
        
        m_series[i] = m
    m_series=m_series+noise  #serie complète ('truth')
    obs_series=m_series[::f] #echantillonage de mesure à une fréquence donnée 
    return (obs_series, m_series, tt_obs)

 
def obs_10 (dt,limt,freq):
    tt=np.arange(0,limt,dt)

    omega = 2*np.pi/(24) 
    temp = 0.2* ( 23+ np.sin(omega*tt) )
    f=freq# en heures pas de temps entre 2 mesures 
    tt_obs=np.arange(0,limt,dt*f)
    tt_obs=tt_obs+15
    m_series = np.empty(len(tt))  # empty array of Nstep values.
    obs_series=np.empty(len(tt_obs))
    
    noise = np.random.normal(0, 0.2, m_series.shape) #bruit du signal pour rajouter erreur de mesure 
    m = 0.

    for i in range(len(tt)):
        m = m + (temp[i] - 0.5 * m)*dt
        m_series[i] = m           
    m_series=m_series+noise
    obs_series=m_series[::f]
    return (obs_series, m_series, tt_obs)
#_______________________________________________________________________________________________________________
# DATA ASSIMILATION 
                    #3DVar
def CostFunction(xin, xb, yo, Pmat,Robs, hobs):
    # Jb
    xx = (xb - xin)/Pmat  
    Jb = 0.5*np.dot(xx,xx)
    # Jo
    innov = yo - hobs@xin    
    Jo = 0.5*np.dot(innov,innov)/Robs
    return Jb+Jo


def CostGrad(xin,xb,Pmat,yobs,Robs):
    # Jb
    gJb = ( np.linalg.inv(Pmat))@(xin - xb)
    # Jo
    m = euler_f(0.5)
    m=m[:,0]
    #xadj = np.zeros_like(x_in)
    iobs = nassim-1
    for i in range(nassim-1):
        innov = -yobs[iobs] + hobs@xin
        gJO=hobs.T@((np.linalg.inv(Robs))@innov)
        
        iobs -= 1
    return gJb + gJo
#________________________________________________________________________________________________