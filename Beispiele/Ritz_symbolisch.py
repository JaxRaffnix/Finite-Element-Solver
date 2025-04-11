# -*- coding: utf-8 -*-
"""
1D FEM Programm 

symbolische Berechnung der Näherungslösung über die Wirkung

-> Globale Ritz-Methode

-d/dx (a d/dx PHI) + b PHI = f



@author: weju0001
"""

import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import string as st
from scipy.optimize import leastsq 

pi=np.pi
x=sy.Symbol('x')

# Die Funktionen in der Dgl.
a=-1
b=0
f=x+1

#Randwerte
xl=0
xr=1


def MakeDic(V,i):
  ret={}
  for j in V:
    ret.update({j:0})
  if i>=0:
    ret[V[i]]=1
  return ret  


# Ansatzfunktion, welche die Randbedingung erfüllt
# phi als funktion der Parameter

# Anzahl der Variablen
Nvar=4
def PHI(p,x):
  #return p[0]*x**2+(1-p[0])*x
  #return p[0]*x**3+p[1]*x**2+(1-p[0]-p[1])*x
  #return p[0]*x**4+p[1]*x**3+p[2]*x**2+(1-p[0]-p[1]-p[2])*x 
  #return p[0]*sy.sin(x)+(1-p[0]*np.sin(1)-p[1])*x+p[1]*x**2
  #return (1+p[0]*(1-sy.cos(1)))/sy.sin(1)*sy.sin(x)+p[0]*(sy.cos(x)-1)
  return x+p[0]*sy.sin(x*sy.pi)+p[1]*sy.sin(2*x*sy.pi)+p[2]*sy.sin(3*x*sy.pi)+p[3]*sy.sin(4*x*sy.pi)

def ExakteLsg(x):
  y=1./6*xx**3+1./2*xx**2+1./3*xx  
  return y



# Variablen + Wirkung
Var=[]
for i in range(Nvar):
  Var+=[sy.Symbol(st.ascii_uppercase[i])]
phi=PHI(Var,x)
dphi=sy.diff(PHI(Var,x),x)
S=sy.integrate(0.5*a*dphi**2+b*phi**2-f*phi,(x,xl,xr))
print "Wirkung: S= ",S

#Ableitung der Wirkung und Erstellen des LGS
dS=range(Nvar)
M=np.zeros((Nvar,Nvar)) 
b=np.zeros(Nvar)
for i in dS:
  dS[i]=sy.diff(S,Var[i])
  # Rechte Sete berechnen alle Var auf null setzen
  b[i]=-dS[i].evalf(subs=MakeDic(Var,-1))
  for j in range(Nvar):
    # Berechne die Matrixelemente, indem nacheinander die 
    # Var[j] auf eins gesetzt werden
    M[i,j]=dS[i].evalf(subs=MakeDic(Var,j))+b[i]
    

print "\nLGS:-----------------"  
print "M= ",M
print "b= ",b  
Xe=np.linalg.solve(M,b)  
  
print "\nLoesung: X= ",Xe

print "\nLoesungsfunktion: phi(x)= ",PHI(list(Xe),x)


# Exakte Lösung und Näherungslösung des Ritz-verfahrens
xx=np.linspace(xl,xr,1000)
ye=ExakteLsg(xx)
ys=np.array([PHI(list(Xe),X) for X in xx])

# Least-Square-Fit
def residual(p,x,data):
  model=x+p[0]*np.sin(x*np.pi)+p[1]*np.sin(2*x*np.pi)+p[2]*np.sin(3*x*np.pi)+p[3]*np.sin(4*x*np.pi)
  return model-data

#out = leastsq(residual, Xe, args=(xx, ye))
#print "Least square y(x)= ",PHI(list(out[0]),x)

# Darstellung
plt.plot(xx,ye,'k',lw=8)
plt.plot(xx,ys,'r',lw=4)
plt.show()

