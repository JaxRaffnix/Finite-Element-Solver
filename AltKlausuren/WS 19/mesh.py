# _____________________________________________________________________________
# Imports


# make top level dir available
import sys
import os
top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, top_level)

import numpy as np
import functions as myfunc
import meshtools as mt
import matplotlib.pyplot as plt


# _____________________________________________________________________________
# Constants


EDGE_LENGTH = 0.01  # custom value

RA = 0.3
DD = 0.01
HM = 0.08
BM = 0.08
MA = 0.01
MB = 0.05
RM0 = 0.095
ZM0 = 0.1
LLS = 0.001
SD = 0.005
SH = 0.01


# _____________________________________________________________________________
# Mesh Generator

# knots, elements = mt.RectangleSegments([0,0], [0.08, 0.08], edge_length=EDGE_LENGTH)

# Randkurve
knots1, elements1 = mt.CircleSegments([0,0], radius=RA, a_min=-np.pi/2, a_max=np.pi/2, edge_length=EDGE_LENGTH)
knots2, elements2 = mt.LineSegments([0, RA], [0, -RA], edge_length=EDGE_LENGTH)
knots, elements = mt.AddSegments(knots1, knots2, closed=True)
 
# R Achse
# knots3, elements3 = mt.LineSegments([0,0], [RM0, 0], edge_length=EDGE_LENGTH)
# knots, elements = mt.AddCurves(knots, elements, knots3, elements3)

# Material
# knots4, elements4 = mt.RectangleSegments([0, -HM/2], [BM, HM/2], edge_length=EDGE_LENGTH)
# knots, elements = mt.AddCurves(knots, elements, knots4, elements4)

# knots, elements = mt.AddSegments(knots, knots3)

mt.PlotBoundary(knots, elements, "Segments")
plt.show()
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE = mt.DoTriMesh(knots, elements, edge_length=EDGE_LENGTH)

print(poi)