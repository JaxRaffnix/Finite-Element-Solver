
# -*- coding: utf-8 -*-

import numpy as np
import meshtools as mt
import matplotlib.pyplot as plt

#########################
#
#  Definition Geometrie
#
#########################

from scipy.constants import epsilon_0 as eps0

length=0.02

RR = 0.06
rk = 20e-3
bs = 4e-3
dalpha=10

eps=1e-5

# Punkte Innenlinien
P3 = [ bs/2 , -np.sqrt(RR**2-(bs/2)**2) ]  #
P4 = [ bs/2 , -np.sqrt(rk**2-(bs/2)**2) ]
P5 = [ bs/2 ,  np.sqrt(rk**2-(bs/2)**2) ]
P6 = [ bs/2 , np.sqrt(RR**2-(bs/2)**2) ]    #
an = np.arctan(2*P5[1]/bs)

#Äusserer Kreis
angles=np.array([-90, -45-dalpha/2, -45+dalpha/2, 135-dalpha/2, 135+dalpha/2, 270])*np.pi/180

# Vordefinierte Punkte
P9 = [RR*np.cos(angles[1]),RR*np.sin(angles[1])]
P10 = [RR*np.cos(angles[2]),RR*np.sin(angles[2])]
P11 = [RR*np.cos(angles[3]),RR*np.sin(angles[3])]
P12 = [RR*np.cos(angles[4]),RR*np.sin(angles[4])]
P13= [ -P6[0], P6[1] ]
P14= [ -P5[0], P5[1] ]
P15= [ -rk, 0 ]
P16= [ -P4[0], P4[1] ]
P17= [ -P3[0], P3[1] ]


#Erstelle Netz
pc1,vc1=mt.CircleSegments([0,0],RR,a_min=angles[0],a_max=angles[1],edge_length=length/10)
for i  in range(4):
  el=length/10
  if i==0 or i==2:
    el/=130
  pci,vci=mt.CircleSegments([0,0],RR,a_min=angles[i+1],a_max=angles[i+2],edge_length=el)
  pc1,vc1=mt.AddSegments(pc1,pci)


p=pc1
v=vc1

# Innere Kreistruktur mit Geraden
#rechte Seite
pi0,vi0=mt.LineSegments( P3 , P4 , edge_length=length/20)
pi1,vi1=mt.CircleSegments([0,0],rk,a_min=-an,a_max=0,edge_length=length/20)
pi2,vi2=mt.CircleSegments([0,0],rk,a_min=0,a_max=an,edge_length=length/20)
pi3,vi3=mt.LineSegments( P5 , P6 , edge_length=length/20)
pii,vii=mt.AddMultipleSegments(pi0,pi1,pi2,pi3)
p,v=mt.AddCurves(p,v,pii,vii,connect=True,connect_points=[P3,P6])

#linke Seite
piii=[ [-w[0],w[1]]  for w in pii ]
pi3,vi3=mt.PointSegments(piii)
p,v=mt.AddCurves(p,v,pi3,vi3,connect=True,connect_points=[P13,P17])

#waagrechte Mittellinie
P7 = [-rk+eps,0]
P8 = [rk-eps,0]
pi4,vi4=mt.LineSegments( P7 , P8 , edge_length=length/20)
p,v=mt.AddCurves(p,v,pi4,vi4)
#p,v=mt.AddCurves(p,v,pi4,vi4,connect=True,connect_points=[P7,P8])


#refine
def myrefine(tri_points, area):
  center_tri = np.sum(np.array(tri_points), axis=0)/3.
  r=np.sqrt(center_tri[0]**2+center_tri[1]**2) 
  if r>0.95*RR:
    max_area=0.005*length**2
  elif np.abs(center_tri[0])<bs/2:
    max_area=0.004*length**2  
  else:
    max_area=0.008*length**2
    
    
  return bool(area>max_area);


poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length,tri_refine=myrefine,writeTo='Netz_SS25')

print(len(poi))

Ps=[P10,P11,P12,P9,P10]
Ps_types = ['Segments','Nodes','Segments','Nodes']
bseg=mt.RetrieveSegments(poi,BouE,li_BE,Ps,Ps_types)

#Zeichne Ränder
for i in range(4):
  mt.PlotBoundary(poi,bseg[i],Ps_types[i])
#Zeichne Innere Linien
mt.PlotBoundary(poi,CuE,' ')
plt.plot(poi[3291,0],poi[3291,1],'o')
plt.show()

G0=bseg[1] +bseg[3]
G0=np.sort(G0)
R0=bseg[0]+bseg[2]

# Knotennummern für f)
Nnodes,dist=mt.FindClosestNode(np.arange(len(poi)),poi,[P13,P14,P15,P16,P17])