#1D heat conduction problem 
import numpy as np
import matplotlib.pyplot as plt
N = 100 #grid points
L = 1 #the length of domain
h = L/(N-1) #grid spacing


T = np.zeros(N) #initialization

#BC at both end
T[0] = 0 
T[-1] = 1

tolerance = 1*np.exp(-6)

numerical_error = 1
iteration = 0

while numerical_error>tolerance:
    T_old = T.copy()

    iteration = iteration+1
    for i in range (1,N-1):
        T[i] = 0.5*(T_old[i-1]+T_old[i+1])
    numerical_error = 0
    for i in range(1, N-1):
        numerical_error = numerical_error + np.abs(T[i]-T_old[i])
x = np.linspace(0,L, N)

plt.plot(x,T)
plt.xlabel('Length')
plt.ylabel('Temperature')
plt.title('1D heat conduction')
plt.show()
