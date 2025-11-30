# IoT proyecto Frontend + Backend

## Ejecucion
Ejecutable mediante docker compose:

```bash
docker compose build # en caso se necesite reconstruir luego de cambios al mqtt.toml, por ejemplo
docker compose up -d # levantar las imagenes docker
docker compose down  # detener las imagenes docker
```

## Desarrollo
Este fue desarrollado en un development shell especifiacdo por `shell.nix`

```bash
nix-shell
```

## Frontend
Frontend simple consistente de un solo archivo html, y se comunica con el backend mediante API calls.

### Idea
Le permite al usuario dibujar en un canvas pixelado. Este dibujo luego sera procesado por el backend, y generara una ruta ciclica de minimo peso. Mas informacion en la seccion del backend.

## Backend
El backend se encarga de generar el grafo, y luego resolver el problema computacional propuesto a continuacion.

Este se encarga ademas, en caso sea activado, de publicar el camino cerrado encontrado en MQTT, segun especificado por el archivo mqtt.toml.

El backend fue desarrollado con Python y FastAPI

### Problema
El problema a resolver es el **chinese postman problem**:

https://en.wikipedia.org/wiki/Chinese_postman_problem

Se trata de encontrar un camino cerrado (bucle) en un grafo ponderado no dirigido que visite todas las aristas al menos UNA vez, a su vez minimizando la cantidad de veces que se repiten aristas y minimizando el peso del camino.

### Algoritmo
El algoritmo para la resolucion del problema esta descrito en la siguiente pagina:

https://www.geeksforgeeks.org/dsa/chinese-postman-route-inspection-set-1-introduction/

### TODO
De ser posible, modificar el algoritmo para que tome en cuenta los angulos entre aristas adyacentes y minimice la suma de los mismos tambien.
