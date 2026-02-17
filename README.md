# FEM 2D Solver

Die allgmeine differenzeille Gleichung

$$
-\frac{\partial}{\partial x}\left(\alpha_1(x, y) \frac{\partial \Phi(x, y)}{\partial x}\right)-\frac{\partial}{\partial y}\left(\alpha_2(x, y) \frac{\partial \Phi(x, y)}{\partial y}\right)+\beta(x, y) \Phi(x, y)=f(x, y)
$$

mit den Dirichlet Randbedingungen

$$
\Phi(x,y) = \delta(x,y) \quad x,y \in G_D
$$

und den Robin Randbedingungen

$$
\left( \alpha_1(x,y) \frac{\partial \Phi(x, y)}{\partial x}, \alpha_2(x,y) \frac{\partial \Phi(x,y)}{\partial y} \right) \cdot \vec{n} + \gamma(x,y) \Phi(x,y) = \rho(x,y)

\\ \quad

x,y \in G_R
$$

wird mithilfe des Galerkin-Verfahrens gelöst.

Beispielanwendung unter [Potentialberchnung](<AltKlausuren/SS 25/potential.ipynb>).

## Example

Anordnung von NL parallelen Platten. Die Platten sind zu zwei Blöcken (rot, blau) parallelgeschaltet. Die Blöcke werden auf konstantem Potential V0 und −V0 gehalten. In den Zwischenräumen befindet sich eine leitfahige, dielektrische Flüssigkeit (σ, r). Die Leitf¨ ahigkeit der Flüssigkeit sei sehr gering, so dass wir folgendes annehmen können:

$$
div(\vec{D}) = \rho
$$

$$
rot(\vec{E}) = \vec{0}
$$

$$
rot(\vec{H}) = \vec{j} + \frac{\partial \vec{D}}{\partial t}
$$

$$
\vec{j} = \sigma \vec{E}
$$

$$
\vec{D} = \epsilon_0 \epsilon_r \vec{E}
$$

Aus den drei obigen Gleichungen folgt, für den Fall einer harmonischen Spannung an den Blöcken, die zu lösende Gleichung:

$$
-\mathrm{div}\left( (\sigma + i\omega\varepsilon_0\varepsilon_r) \mathrm{grad}(\Phi) \right) = 0
$$

![Kondensatoren](<AltKlausuren/WS 18/aufgabe.png>)

![alt text](<AltKlausuren/WS 18/images/solution.png>)

## Meshtools Tipps

### Connect Multiple Segments to one single Polygon

`AddSegments` 

mit optionalem Paramter `closed=True` werden Kanten geschlossen.

### Add Different Polygons to the Figure

`AddCurves`

Wenn 2 Segmente auf-/nebeneinander liegen, muss eines der Segmente mit einem Offset erstellt werden!


### Refine Mesh

` 1 < y < 10 ` is interpreted as `1 < y and y < 10`

Die max area werte müssen klein genug sein, damit sich etwas ändert. z.b. `Edge_length / 100`

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

### Find Boundary Segments

`ROBIN_INDICES[0]` enthält alle elemente des typs `TYPES_LIST[0]` zwischen `BOUNDARY_POINTS[0]` und `BOUNDARY_POINTS[1]`.

`ROBIN_INDICES[1]` enthält alle elemente des typs `TYPES_LIST[1]` zwischen `BOUNDARY_POINTS[1]` und `BOUNDARY_POINTS[2]`.

```python
BOUNDARY_POINTS = [[0,-RA],[RA, 0],[0,RA]]
TYPES_LIST = ["Segments", "Segments"]
ROBIN_INDICES = mt.RetrieveSegments(nodes, boundary_edge, boundary_inidces,BOUNDARY_POINTS, TYPES_LIST)
ROBIN_INDICES = ROBIN_INDICES = np.array(ROBIN_INDICES[0] + ROBIN_INDICES[1])
```