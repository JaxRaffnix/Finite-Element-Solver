# -*- coding: utf-8 -*-
"""
Short Gallery of examples
for meshpy

written by Juergen Weizenecker

"""

import numpy as np
import meshtools as mt
import meshpy.triangle as triangle
import numpy.linalg as la
import matplotlib.pyplot as plt


length=0.3


# Simple mesh rectangle
p,v=mt.RectangleSegments([-1.,-1.],[2.5,3.],edge_length=length)
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length)
mt.PlotMeshNumbers(poi,tri)
print("points, ",poi,flush=True)
print("elements, ",tri,flush=True)
print("Boundary edges",BouE,flush=True)
print("list for boundary edges",li_BE,flush=True)
print("boundary elements",bou_elem,flush=True)


# Use LineSegments
p1,v1=mt.LineSegments([-0.5,0.5],[-1,-1],edge_length=length/5)
p2,v2=mt.LineSegments([-1,-1],[0.,0.5],edge_length=length/5)
p3,v3=mt.LineSegments([0.,0.5],[1,1],edge_length=length/7)
p4,v4=mt.LineSegments([1,1],[-0.5,0.5],edge_length=length/7)
p,v=mt.AddMultipleSegments(p1,p2,p3,p4)
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length)


# simple mesh circle
p,v=mt.CircleSegments([1,2],2,edge_length=length)
mt.DoTriMesh(p,v,edge_length=length)



#
# simple mesh triangle
#
p1,v1=mt.LineSegments([2,2],[-1,-3],edge_length=length)
p2,v2=mt.LineSegments([-1,-3],[3,-1],num_points=10)
p,v=mt.AddSegments(p1,p2,closed=True)
mt.DoTriMesh(p,v,edge_length=length)
#
# rectangle with smooth corners
#
p,v=mt.ORecSegments([1,2],[7,6],0.3,edge_length=length,num_pc=10)
mt.DoTriMesh(p,v,edge_length=length)



# 
# two semicircles
#
p1,v1=mt.CircleSegments([1.,0],1,a_min=-np.pi/2,a_max=np.pi/2,num_points=20)
p2,v2=mt.CircleSegments([1,0],3,a_min=np.pi/2.,a_max=3.*np.pi/2,num_points=20)
p,v=mt.AddSegments(p1,p2,closed=True)
# plot mesh 
mt.DoTriMesh(p,v,edge_length=length)


#
# rectangle and inner circle
#
p1,v1=mt.RectangleSegments([-2,-2],[2.5,3],edge_length=length)
p2,v2=mt.CircleSegments([1,1],1,edge_length=length/5)
p,v=mt.AddCurves(p1,v1,p2,v2)
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length)
print("points, ",poi)
print("elements, ",tri)
print("Boundary Edges",BouE)
print("list boundary edges",li_BE)
print("Inner Curves",CuE)
print("list inner Curve",li_CE)

# boundary nodes -->   use RetrieveSegments for further division of the segments
plt.triplot(poi[:, 0], poi[:, 1], tri)
mt.PlotBoundary(poi,np.array(BouE),'Segments')
mt.PlotBoundary(poi,np.array(CuE),'Nodes')

plt.show()
##############################################################################



#
# rectangle and inner line
#
p1,v1=mt.RectangleSegments([-2,-2],[2.5,3],edge_length=length)
p2,v2=mt.LineSegments([0,0],[1,1],edge_length=length/5)
p3,v3=mt.LineSegments([-1,1],[0,-1],edge_length=length/5)
p,v,indizes=mt.AddMultipleCurves(p1,v1,p2,v2,p3,v3)
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length)
#bou nodes:
plt.triplot(poi[:, 0], poi[:, 1], tri)
mt.PlotBoundary(poi,np.array(CuE),'Segments')
plt.show()



#
# rectangle with holes
p1,v1=mt.LineSegments([-2,-3],[2,-3],num_points=12)
p2,v2=mt.LineSegments([2,3],[-2,3],num_points=12)
p,v=mt.AddSegments(p1,p2,closed=True)
p3,v3=mt.CircleSegments([-0.5,0.5],0.5,edge_length=length)
p,v=mt.AddCurves(p,v,p3,v3)
p4,v4=mt.CircleSegments([1,-1],0.5,edge_length=length)
p,v=mt.AddCurves(p,v,p4,v4)
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length,holes=[(-0.4,0.4),(0.95,-0.8)])

# Find boundary segments between the points below 
Ps=[ [-2,3],[2,-3],[-2,3],[-0.5,0.5],[-0.5,0.5],[1,-1],[1,-1] ]
bseg=mt.RetrieveSegments(poi,BouE,li_BE,Ps,['Nodes','Segments','Nodes','Segments'])
print("bseg",bseg)
plt.triplot(poi[:, 0], poi[:, 1], tri)
mt.PlotBoundary(poi,bseg[0],'Nodes')
mt.PlotBoundary(poi,bseg[1],'Segments')
mt.PlotBoundary(poi,bseg[2],'Nodes')
mt.PlotBoundary(poi,bseg[3],'Segments')
plt.show()



#
# 2D curve
#
t=np.linspace(0,2*np.pi,120)
r=3+np.sin(8*t)
x=r*np.cos(t)
y=r*np.sin(t)
p=[(x[j],y[j]) for j in range(len(t))]
p1,v1=mt.PointSegments(p)
mt.DoTriMesh(p1,v1,edge_length=length)




#
# rectangle and local refinement 
#
p1,v1=mt.RectangleSegments([0,0],[1,1],num_points=100)
p2,v2=mt.RectangleSegments([0.05,0.05],[0.95,0.95],num_points=40)
p,v=mt.AddCurves(p1,v1,p2,v2)
p3,v3=mt.RectangleSegments([0.1,0.1],[0.9,0.9],num_points=20)
p,v=mt.AddCurves(p,v,p3,v3)
mt.DoTriMesh(p,v,edge_length=length)




#
# 2D curve with local mesh refinement I
#
# 
t=np.linspace(0,2*np.pi,120)
r=3+np.sin(8*t)
x=r*np.cos(t)
y=r*np.sin(t)
p=[(x[j],y[j]) for j in range(len(t))]
p1,v1=mt.PointSegments(p)
# function for refinement

def myrefine1(tri_points, area):
  center_tri = np.sum(np.array(tri_points), axis=0)/3.
  x=center_tri[0]
  y=center_tri[1]
  if x>0:
    max_area=0.05*(1+3*x)
  else:
    max_area=0.05
  return bool(area>max_area)

mt.DoTriMesh(p1,v1,tri_refine=myrefine1)






#
# 2D curve with local refinement II
# !! 2 plots
#
# take p1 from above
p2,v2=mt.CircleSegments([0,0],1,edge_length=0.05)
p,v=mt.AddCurves(p1,v1,p2,v2)
# function for refinement
def myrefine2(tri_points, area):
  center_tri = np.sum(np.array(tri_points), axis=0)/3.
  r=np.sqrt(center_tri[0]**2+center_tri[1]**2) 
  max_area=0.3+(0.01-0.3)/(1+0.5*(r-1)**2)
  return bool(area>max_area);
mt.DoTriMesh(p1,v1,tri_refine=myrefine2)  
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,tri_refine=myrefine2)


# find an inner curve in the mesh
# first with Find CurveSegments
def InnerCurve(X):
  eps=1e-5  
  r=np.sqrt((X[0]-0)**2+(X[1]-0)**2)
  if abs(r-1)<eps and X[1]>X[0]:
    return True
  else:
    return False  

plt.triplot(poi[:, 0], poi[:, 1], tri)
curve_nodes,curve_seg,curve_list=mt.FindCurveSegments(poi,tri,InnerCurve,[[0,0]])
# point [0,0] will determine the orientation
mt.PlotBoundary(poi,np.array(curve_seg),'Curve')
# or with boundary segments from DoTriMesh and RetrieveSegments
Ps=[ [1,0],[0,-1] ]
bseg=mt.RetrieveSegments(poi,CuE,li_CE,Ps,['Segments'])
mt.PlotBoundary(poi,bseg[0],'Segments')
plt.show()




#
# 2D curve with local refinement III
# 
#
# take p1 from above
nodes=range(len(p1))
# define tree to speed up node search
from scipy.spatial import cKDTree
p1tree=cKDTree(np.array(p1))
# function for refinement
def myrefine3(tri_points, area):
  center_tri = np.sum(np.array(tri_points), axis=0)/3.
  p0=[(center_tri[0],center_tri[1])]
  node,r=mt.FindClosestNode(nodes,[],p0,tree=p1tree)
  r=r[0]
  max_area=0.3+(0.01-0.3)/(1+r**2) 
  return bool(area>max_area);
mt.DoTriMesh(p1,v1,tri_refine=myrefine3) 



#
# Example for using directly triangle
#

def round_trip_connect(start, end):
  return [(i, i+1) for i in range(start, end)] + [(end, start)]

points = [(1,0),(1,1),(-1,1),(-1,-1),(1,-1),(1,0)]
facets = round_trip_connect(0, len(points)-1)

circ_start = len(points)
points.extend(
        (3 * np.cos(angle), 3 * np.sin(angle))
        for angle in np.linspace(0, 2*np.pi, 29, endpoint=False))

facets.extend(round_trip_connect(circ_start, len(points)-1))

def needs_refinement(vertices, area):
    bary = np.sum(np.array(vertices), axis=0)/3
    max_area = 0.01 + abs((la.norm(bary, np.inf)-1))*0.1
    return bool(area > max_area)

info = triangle.MeshInfo()
info.set_points(points)
info.set_holes([(0,0)])
info.set_facets(facets)

mesh = triangle.build(info, refinement_func=needs_refinement)
#mesh = triangle.build(info) 

mesh_points = np.array(mesh.points)
mesh_tris = np.array(mesh.elements)

import matplotlib.pyplot as pt
print(mesh_points)
print(mesh_tris)
pt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
pt.show()


# boundary nodes
# rectangle with holes
p1,v1=mt.LineSegments([-2,-3],[2,-3],num_points=12)
p2,v2=mt.LineSegments([2,3],[-2,3],num_points=12)
p,v=mt.AddSegments(p1,p2,closed=True)
p3,v3=mt.CircleSegments([-0.5,0.5],0.5,edge_length=length)
p,v=mt.AddCurves(p,v,p3,v3)
p4,v4=mt.CircleSegments([1,-1],0.5,edge_length=length)
p,v=mt.AddCurves(p,v,p4,v4)
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length,holes=[(-0.4,0.4),(0.95,-0.8)])
#print("boundary",bou)
#print("points:",poi)
#print("elements",tri)
#bou=[tuple(x) for x in bou]
#bouS,bl=mt.SortSegments(bou)
#bouS=mt.CheckSegmentSense(tri,bouS,bl)
#print("sorted boundary",bouS)
#print("start segments",bl)



# Simple mesh rectangle with second order points
p,v=mt.RectangleSegments([-1.,-1.],[2.5,3.],edge_length=length)
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length,order=2,show=None)

plt.triplot(poi[:, 0], poi[:, 1], tri[:,0:3])
maxi=np.max(tri[:,0:3])+1
plt.plot(poi[maxi:,0],poi[maxi:,1],'k*')
mt.PlotBoundary(poi,np.array(BouE),'Segments') 
plt.show()
print("points:",poi)
print("elements",tri)
print("boundary",BouE)

  
# connect mesh

# mesh A
p1,v1=mt.LineSegments([0,1],[0,0],edge_length=length)
p2,v2=mt.LineSegments([0,0],[1,0],edge_length=length)
p,v=mt.AddSegments(p1,p2)
p1,v1=mt.CircleSegments([0,0],1,a_min=0,a_max=np.pi/2,edge_length=length)
p,v=mt.AddSegments(p,p1)
pA,tA,bA,lA,bou_elemA,cuA,lcA=mt.DoTriMesh(p,v,edge_length=length)
#mesh B
p1,v1=mt.CircleSegments([0,0],1,a_min=0,a_max=np.pi/2,edge_length=length)
p2,v2=mt.LineSegments([0,1],[2,2],edge_length=length)
p,v=mt.AddSegments(p1,p2)
p1,v1=mt.CircleSegments([0,0],2*np.sqrt(2),a_min=np.pi/4,a_max=0,edge_length=length)
p,v=mt.AddSegments(p,p1)
p1,v1=mt.LineSegments([2*np.sqrt(2),0],[1,0],edge_length=length)
p,v=mt.AddSegments(p,p1)
pB,tB,bB,lB,bou_elemB,cuB,lcB=mt.DoTriMesh(p,v,edge_length=length)
#connect
p,t,b,bl,idn=mt.ConnectMesh(pA,tA,bA,pB,tB,bB,epsilon=1e-8)
plt.triplot(p[:, 0],p[:, 1],t[:,0:3])
k=[x[0] for x in idn]
plt.plot(p[k,0],p[k,1],'ro',mfc='none')

mt.PlotBoundary(p,np.array(b),'Segments')
plt.show()

Ps=[ [1,0],[1,0] ]
bseg=mt.RetrieveSegments(p,b,bl,Ps,['Nodes'])
mt.PlotBoundary(p,bseg[0],'Nodes')

plt.show()  
  
  
  
