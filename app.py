import base64
import io
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from flask import Flask, jsonify, redirect, render_template, request, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from PIL import Image as PILImage

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

# ===== Configuración principal =====
# En Docker se recomienda guardar el modelo en /app/model/modelomejoradoval.h5
MODEL_PATH = os.environ.get("MODEL_PATH", "model/modelomejoradoval.weights.h5")
IMG_SIZE = int(os.environ.get("IMG_SIZE", 320))

# Clases originales del dataset de Kaggle
categorias = {
    0: "battery",
    1: "biological",
    2: "brown-glass",
    3: "cardboard",
    4: "clothes",
    5: "green-glass",
    6: "metal",
    7: "paper",
    8: "plastic",
    9: "shoes",
    10: "trash",
    11: "white-glass",
}

conteo_clases = {categoria: 0 for categoria in categorias.values()}
custom_conteo_clases = {}
ultima_imagen_base64 = None
modelo = None
modelo_error = None


def construir_modelo():
    base_model = Xception(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(12, activation='softmax')(x)

    modelo = Model(inputs=base_model.input, outputs=output)
    return modelo


def cargar_modelo():
    global modelo

    model_file = Path(MODEL_PATH)

    if not model_file.exists():
        print(f"No se encontró el modelo en: {MODEL_PATH}")
        return None

    print("Construyendo arquitectura Xception...")
    modelo = construir_modelo()

    print("Cargando pesos del modelo...")
    modelo.load_weights(str(model_file))

    print("Modelo cargado correctamente")
    return modelo


def preparar_imagen(img: PILImage.Image):
    """Convierte la imagen al formato esperado por Xception."""
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def predecir_desde_bytes(img_bytes: bytes):
    """Clasifica una imagen y regresa etiqueta, confianza y probabilidades."""
    global modelo
    if modelo is None:
        cargar_modelo()
    if modelo is None:
        raise RuntimeError(modelo_error or "El modelo no está cargado.")

    img = PILImage.open(io.BytesIO(img_bytes))
    img_array = preparar_imagen(img)
    predicciones = modelo.predict(img_array, verbose=0)[0]
    clase_predicha = int(np.argmax(predicciones))
    etiqueta = categorias.get(clase_predicha, "desconocido")
    confianza = float(predicciones[clase_predicha])
    probabilidades = {
        categorias[i]: float(valor)
        for i, valor in enumerate(predicciones)
        if i in categorias
    }
    conteo_clases[etiqueta] = conteo_clases.get(etiqueta, 0) + 1
    return etiqueta, confianza, probabilidades


# Intentar cargar al iniciar. Si falta el modelo, la web abre y muestra aviso.
cargar_modelo()


@app.route("/")
def index():
    return render_template(
        "index.html",
        categorias=categorias,
        predicted_label=None,
        confidence=None,
        probabilities=None,
        custom_classes=custom_conteo_clases,
        show_customize=False,
        img_data=None,
        model_error=modelo_error,
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    global ultima_imagen_base64
    predicted_label = None
    confidence = None
    probabilities = None
    error = None

    file = request.files.get("file")
    if not file or file.filename == "":
        error = "Selecciona una imagen antes de clasificar."
    else:
        try:
            img_bytes = file.read()
            ultima_imagen_base64 = base64.b64encode(img_bytes).decode("utf-8")
            predicted_label, confidence, probabilities = predecir_desde_bytes(img_bytes)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        categorias=categorias,
        predicted_label=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
        custom_classes=custom_conteo_clases,
        show_customize=False,
        img_data=ultima_imagen_base64,
        model_error=modelo_error,
        error=error,
    )


@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    global ultima_imagen_base64
    predicted_label = None
    confidence = None
    probabilities = None
    error = None

    photo_data = request.form.get("photo", "")
    try:
        if "," in photo_data:
            photo_data = photo_data.split(",", 1)[1]
        img_bytes = base64.b64decode(photo_data)
        ultima_imagen_base64 = base64.b64encode(img_bytes).decode("utf-8")
        predicted_label, confidence, probabilities = predecir_desde_bytes(img_bytes)
    except Exception as exc:
        error = str(exc)

    return render_template(
        "index.html",
        categorias=categorias,
        predicted_label=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
        custom_classes=custom_conteo_clases,
        show_customize=False,
        img_data=ultima_imagen_base64,
        model_error=modelo_error,
        error=error,
    )


@app.route("/customize", methods=["POST"])
def customize():
    predicted_label = request.form.get("predicted_label")
    return render_template(
        "index.html",
        categorias=categorias,
        predicted_label=predicted_label,
        confidence=request.form.get("confidence"),
        probabilities=None,
        custom_classes=custom_conteo_clases,
        show_customize=True,
        img_data=ultima_imagen_base64,
        model_error=modelo_error,
    )


@app.route("/classify_custom_class", methods=["POST"])
def classify_custom_class():
    selected_class = request.form.get("selected_class")
    if selected_class:
        custom_conteo_clases[selected_class] = custom_conteo_clases.get(selected_class, 0) + 1
    return redirect(url_for("index"))


@app.route("/edit_class", methods=["GET", "POST"])
def edit_class():
    global categorias, conteo_clases
    message = None
    if request.method == "POST":
        old_name = request.form.get("old_name", "").strip()
        new_name = request.form.get("new_name", "").strip()
        if old_name and new_name:
            for key, value in categorias.items():
                if value == old_name:
                    categorias[key] = new_name
                    break
            if old_name in conteo_clases:
                conteo_clases[new_name] = conteo_clases.pop(old_name)
            else:
                conteo_clases.setdefault(new_name, 0)
            message = f"Clase actualizada: {old_name} → {new_name}"
    return render_template("advanced.html", categorias=categorias, message=message)


@app.route("/add_class", methods=["GET", "POST"])
def add_class():
    message = None
    if request.method == "POST":
        custom_class = request.form.get("custom_class", "").strip()
        if custom_class:
            custom_conteo_clases.setdefault(custom_class, 0)
            message = f"Clase personalizada agregada: {custom_class}"
    return render_template("custom.html", custom_classes=custom_conteo_clases, message=message)


@app.route("/summary")
def summary():
    labels = list(conteo_clases.keys())
    counts = list(conteo_clases.values())

    fig = go.Figure([go.Bar(x=labels, y=counts, marker_color="orange")])
    fig.update_layout(
        title="Reporte de Recolección",
        xaxis_title="Tipos",
        yaxis_title="Cantidad",
        xaxis_tickangle=-45,
        autosize=True,
        height=560,
    )

    custom_labels = list(custom_conteo_clases.keys())
    custom_counts = list(custom_conteo_clases.values())

    custom_fig = go.Figure([go.Bar(x=custom_labels, y=custom_counts, marker_color="green")])
    custom_fig.update_layout(
        title="Reporte de Clases Personalizadas",
        xaxis_title="Tipos Personalizados",
        yaxis_title="Cantidad",
        xaxis_tickangle=-45,
        autosize=True,
        height=560,
    )

    return render_template(
        "grafico.html",
        graphd=fig.to_json(),
        customgraphd=custom_fig.to_json(),
        model_error=modelo_error,
    )


# Compatibilidad con el nombre usado en tu versión anterior
@app.route("/grafico")
def grafico():
    return redirect(url_for("summary"))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": modelo is not None, "model_path": MODEL_PATH})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
