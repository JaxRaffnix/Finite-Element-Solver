# FEM 2D Learnings

## Offene Fragen

- wiew funktioniert der Refine Mesh, was ist der input `area`?

## Allgemeine Tipps

Hinzufügen der functions Datei aus top level dir:

```python
# make top level dir available
import sys
sys.path.append("../../")
```

## Meshtools

### Connect Multiple Segments to one single Polygon

`AddSegments` 

mit optionalem Paramter `closed=True` werden Kanten geschlossen.

### Add Different Polygons to the Figure

`AddCurves`

Wenn 2 Segmente auf-/nebeneinander liegen, muss eines der Segmente mit einem Offset erstellt werden!


### Refine Mesh

` 1 < y < 10 ` is interpreted as `1 < y and y < 10`

```python
def refine_mesh(triangle, area):
    """
    triangle are the x,y coordiantes of a singel triangle. Each row is a node.
    Area is the are of the triangle.

    The current triangle will be shrinked if the return value is true
    """
    centroid = myfunc._get_centroid(triangle) # [x_M,y_M]

    # refines mesh if x is less than MAX
    if centroid[x] > X_MAX:
        max_area = 0.5
    else:
        max_area = 1

    return bool(area > max_area)
```