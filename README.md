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

Adicionalmente, el frontend establece la conexion con el mosquitto broker mediante websockets. Requisitos a continuacion.

### Websockets a MQTT
La decision de que la comunicacion al broker mosquitto sea mediante el frontend se genero en base a un problema que causaba FastAPI y como bloqueaba la comunicacion con el cliente de MQTT. La solucion resulta en que mosquitto puede aceptar inputs de websockets y pueden ser leidas por otros suscriptores, como si fuese un mensaje de MQTT convencional. La desventaja siendo que hay reduccion de velocidad, pero es mejor eso al problema mencionado previamente.

Para hacer que mosquitto acepte websockets se requiere una configuracion de la siguiente forma:

```bash
allow_anonymous true

listener 1883 0.0.0.0
protocol mqtt

listener 9001
protocol websockets
```

### Configuracion del broker
Utilizamos el archivo `mqtt.json` para controlar los parametros de MQTT:

```json
{
  "host": "192.168.1.39",
  "port": 9001,
  "topic": "laser/frame"
}
```

Cabe destacar que `port` en el json NO es 1883, sino 9001. Como explicado previamente, el frontend se comunica con el broker mosquitto mediante websockets, y para ello utiliza el puerto 9001 de (como se puede observar arriba en la configuracion de mosquitto). El puerto 1883 es para que se pueda establecer conexion usual con clientes MQTT de forma usual, el cual seria el caso de ESP32 por ejemplo.

## Backend
El backend se encarga de generar el grafo, y luego resolver el problema computacional propuesto a continuacion.

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
