import meshpy as mp
import meshtools as mt
import matplotlib.pyplot as plt

LENGTH = 0.1
length = 0.1

p1,v1=mt.LineSegments([-0.5,0.5],[1,1],edge_length=LENGTH/5)
p2,v2=mt.LineSegments([-1,-1],[0.,0.5],edge_length=LENGTH/5)

p,v=mt.AddMultipleSegments(p1,p2)
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=LENGTH)
plt.show()