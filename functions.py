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
#euler forward function 
def euler_f(dx,dt,limt,plot=0):    #euler forward 
    #PARAMETERS
    dX=0.1
    limx=100
    XX=np.arange(0,limx,dX)
    xx=np.arange(0,limx,dx)
    tt=np.arange(0,limt,dt)

#variables 
    s=5  #standard dev   gaussian
    mu=0   #moyenne gaussienne 
    K=0.6

    Tf=np.zeros((len(tt),len(xx)))
    Tf[0,:]=gauss(xx, mu, s)
    
    #Conditions aux limites 
    Tf[:,0]=temp_cycle(tt,dt)
    #Tf[:,-1]=np.zeros(len(tt))

    for i in range(0,len(tt)-1):
        for j in range(1,len(xx)-1):
            Tf[i+1,j]=Tf[i,j]+(K*dt/(dx**2))*(Tf[i,j+1]+Tf[i,j-1]-2*Tf[i,j])
        if plot==1:
            if i%24==0:
                plt.plot(xx,Tf[i,:],label='t='+str(i)+' h')
                plt.legend()
                plt.xlabel('Space domain (cm)')
                plt.ylabel('Température')
                plt.grid()
                plt.title('Heat diffusion')
    return Tf


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