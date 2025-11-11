# Problema
El problema a resolver es el **chinese potsman problem**:
https://en.wikipedia.org/wiki/Chinese_postman_problem

Se trata de encontrar un camino cerrado (bucle) en un grafo ponderado no dirigido que visite todas las aristas al menos UNA vez, a su vez minimizando la cantidad de veces que se repiten aristas y minimizando el peso del camino.

# Algoritmo
Se utilizara el algoritmo de T-joins, descrito tambien por el mismo articulo de wikipedia:
https://en.wikipedia.org/wiki/Chinese_postman_problem

# Ejecucion
Para correr el programa usar:

```bash
nix-shell
python3 main.py
```
