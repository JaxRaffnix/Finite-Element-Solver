# FEM 2D Learnings

## Offene Fragen

- wiew funktioniert der Refine Mesh, was ist der input `area`?

## Allgemeine Tipps

Hinzufügen der functions Datei aus top level dir:
```
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

Wenn 2 Segmente auf-/nebeneinander liegen, muss eines der Segmente mit einem Offset erstellt werden.
