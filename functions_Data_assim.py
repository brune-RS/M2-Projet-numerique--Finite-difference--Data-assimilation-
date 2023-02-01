import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import scipy.optimize as so



#______________________________________________________
def gauss(x,mu,s):
        f=norm.pdf(x,mu,s)
        return f
def integral(Uin,dx):
     return np.sum(Uin*dx) 
    
def RMSE(exact_solution, num_solution):

    mean_of_differences_squared = ((exact_solution- num_solution)**2).mean() 

    rmse_val = np.sqrt(mean_of_differences_squared)         

    return rmse_val          



#________________________________________________________________________________________
    #DATA ASSIMILATION
#______________________________________________________________________________________




#create observation files at 5 cm and 10 cm depth 

def obs_5 (dt,limt,freq):  #frequency ( minutes  )
    tt=np.arange(0,limt,dt)
    omega = 2*math.pi/(24) # Period # mabye add stochastic value to change length of period
    temp = 0.5* ( 13+ np.sin(omega*tt) ) #create temperature oscillations (diurnal cycle)
    f=freq # en heures pas de temps entre 2 mesures 
    
    tt_obs=np.arange(0,limt,dt*f) # times where observations are available
    tt_obs=tt_obs+10  #shift in time (lag between what happens at surface and deeper)
    
    m_series = np.empty(len(tt))  
    obs_series=np.empty(len(tt_obs))
    
    noise = np.random.normal(0, 0.25, m_series.shape) #signal noise (measurement errors + variability)
    m = 0.

    for i in range(len(tt)):
        m = m + (temp[i] - 0.5 * m)*dt
        m_series[i] = m
    m_series=m_series+noise  # complete serie ('truth')
    obs_series=m_series[::f] #sampling of measurements at a given frequency
    return (obs_series, m_series, tt_obs)
    
 
 
def obs_10 (dt,limt,freq):
    tt=np.arange(0,limt,dt)
    omega = 2*math.pi/(24) # Period # mabye add stochastic value to change length of period
    temp = 0.2* ( 23+ np.sin(omega*tt) ) #create temperature oscillations (diurnal cycle)
    f=freq # en heures pas de temps entre 2 mesures 
    
    tt_obs=np.arange(0,limt,dt*f) # times where observations are available
    tt_obs=tt_obs+15  #shift in time (lag between what happens at surface and deeper)
    
    m_series = np.empty(len(tt))  
    obs_series=np.empty(len(tt_obs))
    
    noise = np.random.normal(0, 0.2, m_series.shape) #signal noise (measurement errors + variability)
    m = 0.

    for i in range(len(tt)):
        m = m + (temp[i] - 0.5 * m)*dt
        m_series[i] = m
    m_series=m_series+noise  # complete serie ('truth')
    obs_series=m_series[::f] #sampling of measurements at a given frequency
    return (obs_series, m_series, tt_obs)







#_______________________________________________________________________
def euler_f_BC_periodic(dx,dt,limx,limt,K,plot=0):    #euler forward  with realistic boundary conditions at the surface
    #PARAMETERS

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
    Tf[0,:]=gauss(xx, mu, s)#initial condition     
    #Boundary conditions 
    Tf[:,0]=temp_cycle(tt,dt)
    

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



def temp_cycle (tt,dt):

    omega = 2*np.pi/(24) # period (minutes)
    temp = 5 * ( 1.5 + np.sin(omega*tt) ) #amplitude 

    m_series = np.empty(len(tt)) 
    m = 0.

    for i in range(len(tt)):
        m = m + (temp[i] - 0.5 * m)*dt
        m_series[i] = m
    return m_series
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