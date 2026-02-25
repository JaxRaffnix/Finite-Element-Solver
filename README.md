# 2D Finite Elemente Analyse

Dieses Projekt entschand im Rahmen der Vorlesung "Methoden der Feldberechnung" bei Prof. Jürgen Weizenecker an der Hochschule Karslsruhe. Der Meshgenerator wurde bereitgestellt und ist eine Erweiterung von [meshpy](https://documen.tician.de/meshpy/). Die Fintive-Elemente-Berechnung mithilfe des Galerkin-Verfahrens wurde eigenständig implementiert. 

Ziel ist es, bekannte Randwertaufgaben für verschiedene Anordnungen von elektrisch leitende Materialien zu lösen.

## Theorie

Die allgmeine differenzielle Gleichung

$$
-\frac{\partial}{\partial x}\left(\alpha_1(x, y) \frac{\partial \Phi(x, y)}{\partial x}\right)-\frac{\partial}{\partial y}\left(\alpha_2(x, y) \frac{\partial \Phi(x, y)}{\partial y}\right)+\beta(x, y) \Phi(x, y)=f(x, y)
$$

wird diskretisiert und numerisch berechnet. Dabei ist $\Phi(x, y)$ die unbekannte Zielfunktion, $\alpha_1(x, y)$ und $\alpha_2(x, y)$ sind die Diffusionskoeffizienten, $\beta(x, y)$ ist der Reaktionskoeffizient und $f(x, y)$ ist die Quellfunktion.

Über die Lösung sind folgende Randbedingungen bekannt:

### Dirichlet Randbedingungen

Der Wert der Funktion $\Phi(x,y)$ ist an allen Punkten auf dem Rand $G_D$ bekannt und wird durch die Funktion $\delta(x,y)$ beschrieben:

$$
\Phi(x,y) = \delta(x,y) \quad x,y \in G_D
$$

### Robin Randbedingungen

Für alle Punkte auf dem Rand $G_R$ gilt die folgende Einschränkung mit den bekannten Funktionen $\alpha_1(x,y)$, $\alpha_2(x,y)$, $\gamma(x,y)$ und $\rho(x,y)$:

$$
\left( \alpha_1(x,y) \frac{\partial \Phi(x, y)}{\partial x}, \alpha_2(x,y) \frac{\partial \Phi(x,y)}{\partial y} \right) \cdot \vec{n} + \gamma(x,y) \Phi(x,y) = \rho(x,y)
\quad
x,y \in G_R
$$

## Beispiel

![Aufgabe](<AltKlausuren/SS 25/Ansicht.png>)

Für die gzeigte Materialanordung wird der Stromfluss untersucht, wenn die Spannung $V_0$ und $-V_0$ an den Anschlüssen angelegt wird. Die Materialien sind dielektrisch und leitfähig, somit gilt:

$$
\operatorname{div}\left(\vec{j}_\sigma+\vec{j}_D\right)=\operatorname{div}\left(\vec{j}+\frac{\partial \vec{D}}{\partial t}\right)=\operatorname{div}(\operatorname{rot} \vec{H})=0
$$

wobei für die Stromdichte $\vec{j}=\sigma \vec{E}=-\sigma$ grad $\Phi$ gelte. Damit ergibt sich das harmonische Problem

$$
-\operatorname{div}\left(\left(\sigma+i \varepsilon_0 \varepsilon_r \omega\right) \operatorname{grad} \Phi(x, y)\right)=0
$$

Die vollständige Beschreibung aller Parameter sowie die Berechnung findet sich im [folder](<AltKlausuren/SS 25>).

### Ergebnisse

| Beschreibung | Bild |
|---|---|
| Mesh | ![Mesh](<AltKlausuren/SS 25/mesh.png>) |
| Potentiallinien | ![Potentiallinien](<AltKlausuren/SS 25/potential.png>) |
| Stromlinien | ![Stromlinien](<AltKlausuren/SS 25/current.png>) |

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