# Recicla IA - Clasificador de residuos con Flask + Docker

Este proyecto despliega una web para clasificar residuos con un modelo entrenado en Keras/TensorFlow basado en Xception.

## 1. Estructura

```text
residuos_ai_docker_final/
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── model/
│   └── modelomejoradoval.h5   # aquí debes colocar tu modelo real
├── templates/
└── static/
```

## 2. Exportar modelo desde Google Colab

Al final de tu notebook de entrenamiento ejecuta:

```python
model.save('/content/modelomejoradoval.h5')
from google.colab import files
files.download('/content/modelomejoradoval.h5')
```

Si tu variable del modelo tiene otro nombre, cambia `model` por el nombre correcto. Por ejemplo:

```python
modelomejoradoval.save('/content/modelomejoradoval.h5')
```

## 3. Colocar modelo

Copia el archivo descargado dentro de:

```text
model/modelomejoradoval.h5
```

## 4. Levantar con Docker

```bash
docker compose up --build
```

Abre:

```text
http://localhost:5000
```

## 5. Comandos útiles

Ver estado:

```bash
curl http://localhost:5000/health
```

Apagar:

```bash
docker compose down
```

## 6. Subir a Render, Railway o VPS

Para alojarlo en línea, sube este proyecto a GitHub. En el servidor debes conservar la variable:

```text
MODEL_PATH=/app/model/modelomejoradoval.h5
```

Si la plataforma no permite subir archivos grandes al repositorio, guarda el modelo en un almacenamiento externo o súbelo como volumen/archivo privado.
