import numpy as np
import matplotlib.pyplot as plt

N = 100 #number of nodes
L = 1 #Length of the domain
h = L/(N-1) #grid spacing 

#bondary_condition
T=np.zeros(N)
T[0] = 0
T[-1]= 1

k = 340 #thermal conductivity
A = 0.00001 #cross section

tolerance = 1*np.exp(-8)
num_erro = 1
iteration = 0

while num_erro>tolerance:
    iteration = iteration+1
    T_old = T.copy()
    for i in range (1, N-1):
        a_w = (k*A)/h
        a_e = (k*A)/h
        if i==0:
            a_w = 0
        a_p = a_e+a_w
        T[i] = (a_e*T[i+1]+a_w*T[i-1])/a_p

    num_erro=0
    for i in range(1, N-1):
        num_erro = num_erro+np.abs(T[i]-T_old[i])

x= np.linspace(0,L,N)
plt.plot(x,T)
plt.xlabel('length')
plt.ylabel('temperature')
plt.title('Generalized 1D heat conduction')
plt.show()    


