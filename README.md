# Recomendation system
Calculos para generar recomendaciones a clientes segun comportamiento similar de otros clientes.
Para un uso de memoria y tiempo eficiente, se utiliza estrucutra de datos disperso y utilizacion de multiples hilos para reducir el tiempo y costo de memoria.


## Requerimientos

- [Python >= 3.6](https://www.python.org/)
- [Pipenv](https://github.com/pypa/pipenv)

## Instalación

1. Clone repository: `git clone https://github.com/Joaquinecc/api-bristol.git`
2. Install dependencies: `pipenv install`
3. Activate virtualenv: `pipenv shell`
4. Create a file called `settings-params.json` in root directory
5. Insert the following lines into the file:

```
{
    "read_path":<nput file>,
    "path_write":<nombre del archivo donde se almacenara los resultados>,
    "chunksize":10000,
    "topNProduct":15,
    "topNSimilarity":30
}
```
6. Run script: `python main.py`

## Guia de uso


La variable **read_path** indica la dirección del archivo de entrada. Este archivo de entrada es un archivo csv, donde cada fila representa un cliente y cada columna un producto o familia. El valor de la celda contiene el rating que el cliente tiene con respecto al producto o familia.

**topNProduct**: Cuantos recomendaciones generar para cada cliente

**topNSimilarity**: Con cuantos clientes se hace la comparación. (Se eligen primero los de mayor similitud)

**chunksize** : El archivo de entrada es muy grande, por lo cual se lee en pedazos. Con esta variable se indica de a cuanto se quiere leer, por defecto es 10000
