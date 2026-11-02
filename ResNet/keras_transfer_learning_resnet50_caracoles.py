# -*- coding: utf-8 -*-
"""Clasificación de familias de caracoles con ResNet50 (transfer learning).

Se entrena el mismo modelo dos veces: una con las imágenes originales y otra con
aumentación de datos (Albumentations). Ambas usan la misma partición 70/15/15; la
prueba se reserva para la evaluación final.

Las imágenes SEM están en escala de grises: se cargan con un canal, se reducen a
224x377 conservando la proporción original (712x423, sin estirar ni rellenar) y el
canal gris se duplica a 3 canales idénticos (R=G=B). Dentro del modelo (capa
Rescaling) se resta el mismo valor (114.8) a los tres canales para llevarlos a la
escala de ImageNet sin desbalancear los grises (no se usa preprocess_input).
"""

import os
import json
import shutil
import time
import datetime
import zoneinfo

# Límite de memoria temporal por convolución de MIOpen (ROCm). El valor por defecto (4 GB)
# se queda corto con imágenes grandes; se sube a 8 GB por seguridad (la GPU tiene 64 GB).
# Debe definirse antes de importar TensorFlow.
os.environ.setdefault("TF_CUDNN_WORKSPACE_LIMIT_IN_MB", "8192")

import albumentations as A
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score
)

# 1. CONFIGURACIÓN
# Semilla fija para que la partición y la inicialización se repitan igual en cada corrida.
SEED = 42
tf.keras.utils.set_random_seed(SEED)

NOMBRE_MODELO = "resnet50"
folder_path = '/lustre/home/estudiante_02/TransferLearning/CARACOLES'
# Carpeta donde se guardan gráficas y modelos.
DIR_RESULTADOS = '.'
# En el script las figuras solo se guardan; en el notebook además se muestran.
MOSTRAR_FIGURAS = False
# (alto, ancho) en el orden de Keras. Alto 224 como en el preentrenamiento de ImageNet y
# ancho 377 para conservar la proporción original 712x423 (no se estira ni se rellena).
IMG_SIZE = (224, 377)
BATCH_SIZE = 32

# Fase 1: solo se entrena la cabecera nueva (pesos aleatorios), admite una LR alta.
EPOCAS_FASE1 = 15
LR_FASE1 = 1e-3
# Fase 2: LR 100 veces menor para ajustar los pesos de ImageNet sin borrarlos.
EPOCAS_FASE2 = 15
LR_FASE2 = 1e-5
# Épocas sin mejora en val_loss antes de detener el entrenamiento.
PACIENCIA = 4

# Exporta también un .tflite (pesos en float16, ~mitad de tamaño) para la página web con LiteRT.js.
EXPORTAR_LITERT = True

os.makedirs(DIR_RESULTADOS, exist_ok=True)


def ruta_salida(nombre_archivo):
    return os.path.join(DIR_RESULTADOS, nombre_archivo)


def guardar_figura(fig, nombre_archivo):
    fig.savefig(ruta_salida(nombre_archivo), dpi=300, bbox_inches='tight')
    if MOSTRAR_FIGURAS:
        plt.show()
    plt.close(fig)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO HPC] GPUs detectadas: {len(gpus)} -> {gpus}", flush=True)
else:
    print("[WARN HPC] No se detectó GPU. El entrenamiento correrá en CPU.", flush=True)

tiempo_inicial = time.time()
zona_hermosillo = zoneinfo.ZoneInfo("America/Hermosillo")

# 2. CARGA DE IMÁGENES Y DISTRIBUCIÓN DE CLASES
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"No se encontró el directorio del dataset en: {folder_path}")

# Se cargan sin lotes y sin mezclar para conservar el orden y dividir por índices.
# color_mode='grayscale': las imágenes SEM tienen un solo canal.
# 712x423 -> 377x224: misma escala en ambos ejes, la concha conserva su forma.
# interpolation='area' es la más adecuada para reducir: evita el dentado y conserva la textura.
dataset_completo = keras.utils.image_dataset_from_directory(
    folder_path,
    label_mode='int',
    color_mode='grayscale',
    image_size=IMG_SIZE,
    interpolation='area',
    batch_size=None,
    shuffle=False
)

class_names = dataset_completo.class_names
num_clases = len(class_names)

# Paso todo a arreglos de numpy para poder partir por índices con scikit-learn.
# Se guardan en uint8 (0-255), que es el formato que espera Albumentations y ocupa 4 veces menos memoria.
imagenes_totales = np.empty((len(dataset_completo), *IMG_SIZE, 1), dtype=np.uint8)
etiquetas_totales = np.empty(len(dataset_completo), dtype=np.int32)
for i, (img, label) in enumerate(dataset_completo):
    imagenes_totales[i] = np.clip(np.round(img.numpy()), 0, 255).astype(np.uint8)
    etiquetas_totales[i] = label.numpy()
del dataset_completo

print(f"\nTotal de imágenes cargadas: {len(imagenes_totales)}")
print(f"Número de familias detectadas: {num_clases}\n")
for idx, familia in enumerate(class_names):
    print(f"  [{idx:02d}] {familia}")

conteos_por_clase = np.bincount(etiquetas_totales, minlength=num_clases)

fig, ax = plt.subplots(figsize=(10, 5))
colores = plt.cm.viridis(np.linspace(0, 0.85, num_clases))
barras = ax.bar(class_names, conteos_por_clase, color=colores, edgecolor='black')
for barra in barras:
    yval = barra.get_height()
    ax.text(barra.get_x() + barra.get_width() / 2.0, yval + conteos_por_clase.max() * 0.015,
            f'{yval}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title("Distribución de Familias de Caracoles", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Familias de caracoles", fontsize=12)
ax.set_ylabel("Cantidad de imágenes", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
guardar_figura(fig, f'distribucion_caracoles_{NOMBRE_MODELO}.png')

# 3. PARTICIÓN ESTRATIFICADA (70/15/15) Y PESOS DE CLASE
minimo_por_clase = int(conteos_por_clase.min())
if minimo_por_clase < 5:
    clase_pequena = class_names[int(np.argmin(conteos_por_clase))]
    raise ValueError(
        f"La clase {clase_pequena!r} tiene {minimo_por_clase} imágenes. "
        "Se requieren al menos cinco para hacer las dos divisiones estratificadas."
    )

indices = np.arange(len(etiquetas_totales))

# Primero 70% entrenamiento y 30% temporal; luego el temporal se parte en 15/15.
# stratify conserva la proporción de cada familia en las tres particiones.
train_idx, temp_idx, y_train, y_temp = train_test_split(
    indices, etiquetas_totales, test_size=0.30, stratify=etiquetas_totales, random_state=SEED
)
val_idx, test_idx, y_val, y_test = train_test_split(
    temp_idx, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

x_train = imagenes_totales[train_idx]
x_val   = imagenes_totales[val_idx]
x_test  = imagenes_totales[test_idx]
del imagenes_totales

print(f"\nImágenes por partición -> Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}\n")
print("Desglose de ejemplares por familia:")
print("-" * 55)
for i, nombre in enumerate(class_names):
    print(f"  {nombre:<16} | Train: {np.sum(y_train == i):3d} | Val: {np.sum(y_val == i):3d} | Test: {np.sum(y_test == i):3d}")
print("-" * 55)

# 'balanced' da más peso a las familias con pocas imágenes (Risooinidae) para que la red
# no las ignore. Se calculan solo con entrenamiento: validación y prueba no deben influir.
clases_unicas = np.unique(y_train)
pesos_sklearn = compute_class_weight(class_weight='balanced', classes=clases_unicas, y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(clases_unicas, pesos_sklearn)}

print("\nPesos de clase (compute_class_weight 'balanced'):")
for i, nombre in enumerate(class_names):
    print(f"  [{i}] {nombre:<16}: {class_weight_dict.get(i, 1.0):.4f}")

# 4. AUMENTO DE DATOS (ALBUMENTATIONS) Y PIPELINES tf.data
# Pipeline enfocado en alta calidad. No se usan volteos (flips): reflejar una concha invierte
# el sentido del enrollamiento (dextral/sinistral), un rasgo taxonómico real.
train_transform = A.Compose([
    # Rotación leve; INTER_CUBIC mantiene la nitidez de la textura y
    # BORDER_REFLECT_101 es el relleno que menos bordes artificiales deja.
    A.Rotate(limit=15, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    # Traslación de hasta 5 % sin escalado ni rotación (en Albumentations 2.x ShiftScaleRotate
    # está obsoleto y se reemplaza por Affine).
    A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, scale=1.0, rotate=0,
             interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    # Ruido sutil para que el modelo ignore imperfecciones del fondo del SEM.
    # Equivale a var_limit=(10, 30) de versiones anteriores: desviación de sqrt(10)/255 a sqrt(30)/255.
    A.GaussNoise(std_range=(0.0124, 0.0215), p=0.2),
], seed=SEED)


def aplicar_albumentations(imagen):
    return train_transform(image=imagen)["image"]


def aumentar(imagen, etiqueta):
    # Albumentations trabaja con NumPy, por eso se envuelve en tf.numpy_function.
    imagen = tf.numpy_function(aplicar_albumentations, [imagen], tf.uint8)
    imagen.set_shape((*IMG_SIZE, 1))
    return imagen, etiqueta


def prepare_sem_image(image_gray, label):
    """image_gray tiene forma (H, W, 1) y valores en [0, 255]."""
    # 1. Duplicar el canal gris en 3 canales idénticos (R=G=B) para conservar
    #    los pesos de la primera capa convolucional preentrenada en ImageNet.
    image_rgb = tf.image.grayscale_to_rgb(image_gray)
    # 2. Se deja en float32 con valores 0-255: la normalización está dentro del modelo,
    #    así el .tflite recibe la imagen tal cual y la web no tiene que normalizar.
    return tf.cast(image_rgb, tf.float32), label


# shuffle mezcla el orden en cada época; prefetch prepara el siguiente lote mientras la GPU entrena.
dataset_train = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=len(x_train), seed=SEED)
    .map(prepare_sem_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
# El aumento se aplica solo al entrenamiento y después de la partición.
dataset_train_aug = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=len(x_train), seed=SEED)
    .map(aumentar, num_parallel_calls=tf.data.AUTOTUNE)
    .map(prepare_sem_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
# Validación y prueba no se mezclan ni se aumentan: se evalúan tal cual.
dataset_validacion = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
                      .map(prepare_sem_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
dataset_prueba     = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                      .map(prepare_sem_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# 5. VISUALIZACIÓN: IMÁGENES ORIGINALES VS. AUMENTADAS
muestra_imgs = x_train[:9]
muestra_lbls = y_train[:9]
imgs_aumentadas = np.stack([aplicar_albumentations(img) for img in muestra_imgs])

for imgs, etiqueta, archivo in [
    (muestra_imgs, "Original", "puras"),
    (imgs_aumentadas, "Aumentada", "aumentadas"),
]:
    fig = plt.figure(figsize=(12, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[i].squeeze(), cmap="gray", vmin=0, vmax=255)
        plt.title(f"{etiqueta}: {class_names[muestra_lbls[i]]}", fontsize=10, fontweight='bold')
        plt.axis("off")
    plt.suptitle(f"Muestras de entrenamiento ({etiqueta.lower()}s)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    guardar_figura(fig, f'{NOMBRE_MODELO}_muestra_imagenes_{archivo}.png')

# 6. ARQUITECTURA: RESNET50 PREENTRENADA + CABECERA
def construir_modelo(usar_aug):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    # Congelada al inicio: en la fase 1 solo aprende la cabecera nueva.
    base_model.trainable = False

    # Entrada: gris replicado en 3 canales, valores 0-255. El aumento se hace en tf.data, no en el modelo.
    entradas = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    # Normalización dentro del modelo: restar el mismo valor (114.8, promedio de las medias de
    # ImageNet) a los 3 canales. Así los números quedan en la escala que esperan las
    # BatchNormalization congeladas y la imagen sigue siendo gris (R=G=B). No se usa
    # preprocess_input: restaría medias distintas por canal y desbalancearía los grises.
    x = layers.Rescaling(1.0, offset=-114.8, name="normalizacion")(entradas)
    # training=False fija las BatchNormalization de la base en modo inferencia,
    # también durante el ajuste fino, para no destruir sus estadísticas de ImageNet.
    x = base_model(x, training=False)

    # Global average pooling resume cada mapa en un número: evita una capa densa enorme.
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    # Dropout apaga el 40% de las neuronas al azar en cada paso para que no se memorice el entrenamiento.
    x = layers.Dropout(0.4, seed=SEED, name="dropout")(x)
    # softmax convierte las salidas en una probabilidad por familia que suma 1.
    salidas = layers.Dense(num_clases, activation="softmax", name="clasificador_familias")(x)

    modelo = keras.Model(entradas, salidas, name=f"ResNet50_{'con' if usar_aug else 'sin'}_aug")
    return modelo, base_model


def descongelar_capas_profundas(base_model):
    """Libera solo conv5; conv1-conv4 y todas las BatchNormalization quedan congeladas.

    Los primeros bloques detectan bordes y texturas genéricas que sirven para
    cualquier imagen; el último es el más específico y el que conviene adaptar a
    las conchas. Con ~2,000 imágenes de entrenamiento, liberar más bloques solo
    aumenta la capacidad de memorizar.
    """
    base_model.trainable = True
    for layer in base_model.layers:
        es_ultimo_bloque = layer.name.startswith("conv5_")
        es_bn = isinstance(layer, layers.BatchNormalization)
        layer.trainable = es_ultimo_bloque and not es_bn
    return sum(layer.trainable for layer in base_model.layers)

# 7. ENTRENAMIENTO EN DOS FASES
# Si la pérdida de validación no mejora en PACIENCIA épocas, se detiene y
# se recuperan los pesos de la mejor época (evita seguir sobreajustando).
def early_stopping():
    return keras.callbacks.EarlyStopping(monitor='val_loss', patience=PACIENCIA, restore_best_weights=True, verbose=1)


# Si val_loss no mejora en 2 épocas, la LR se reduce a la mitad: suaviza los picos de las curvas.
def reducir_lr():
    return keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)


def exportar_litert(modelo, nombre_experimento):
    """Convierte el modelo a .tflite para usarlo en el navegador con LiteRT.js.

    Entrada: float32 de forma [1, 224, 377, 3]: imagen en gris reducida a 377x224,
    con valores 0-255 (sin normalizar) y el canal replicado 3 veces.
    Salida: una probabilidad por familia, en el orden de clases_*.json.
    """
    # El aumento está en tf.data, así que el modelo ya es directamente de inferencia.
    dir_temporal = ruta_salida(f"savedmodel_{NOMBRE_MODELO}_{nombre_experimento}")
    # Batch fijo en 1: LiteRT.js (web) no admite dimensiones dinámicas en la entrada.
    modelo.export(dir_temporal, format="tf_saved_model", verbose=False,
                  input_signature=[tf.TensorSpec([1, *IMG_SIZE, 3], tf.float32, name="imagen_sem")])
    convertidor = tf.lite.TFLiteConverter.from_saved_model(dir_temporal)
    # float16 reduce el archivo a la mitad sin pérdida de exactitud apreciable.
    convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
    convertidor.target_spec.supported_types = [tf.float16]
    contenido = convertidor.convert()
    shutil.rmtree(dir_temporal)
    ruta_tflite = ruta_salida(f"modelo_{NOMBRE_MODELO}_{nombre_experimento}.tflite")
    with open(ruta_tflite, "wb") as archivo:
        archivo.write(contenido)
    print(f"Modelo LiteRT guardado en: {ruta_tflite} ({len(contenido) / 1e6:.1f} MB)")


def entrenar_modelo(usar_aug, nombre_experimento):
    print("\n" + "=" * 75)
    print(f" ENTRENANDO: {nombre_experimento.upper()}")
    print("=" * 75)

    # Reiniciar sesión y semilla para que los dos modelos partan exactamente igual.
    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    modelo, base_model = construir_modelo(usar_aug)
    # Entropía cruzada "sparse" porque las etiquetas son enteros (0..5), no one-hot.
    criterio_loss = keras.losses.SparseCategoricalCrossentropy()

    # Fase 1: la cabecera empieza con pesos aleatorios; si se afinara toda la red desde
    # ya, sus gradientes grandes estropearían los pesos de ImageNet. Primero se adapta sola.
    print("\nFase 1: entrenamiento de la cabecera (base congelada)")
    # clipnorm=1.0 limita el tamaño de cada actualización de pesos: evita saltos bruscos (picos).
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_FASE1, clipnorm=1.0),
                   loss=criterio_loss, metrics=['accuracy'])
    modelo.summary(show_trainable=True)
    # Mismo conjunto de entrenamiento; solo cambia si se aplica Albumentations.
    datos_entrenamiento = dataset_train_aug if usar_aug else dataset_train
    hist_f1 = modelo.fit(
        datos_entrenamiento,
        validation_data=dataset_validacion,
        epochs=EPOCAS_FASE1,
        class_weight=class_weight_dict,
        callbacks=[early_stopping(), reducir_lr()],
        verbose=2
    )
    # Se guarda en qué época terminó la fase 1 para marcarla en las gráficas.
    epoca_fin_fase1 = len(hist_f1.history['loss'])

    # Fase 2: con la cabecera ya estable, se deja ajustar el último bloque
    # (el más específico) para que se adapte a las conchas.
    print("\nFase 2: ajuste fino del último bloque (conv5)")
    capas_entrenables = descongelar_capas_profundas(base_model)
    print(f"Capas entrenables de ResNet50: {capas_entrenables}")
    # Recompilar es obligatorio tras cambiar qué capas son entrenables. LR baja
    # para ajustar los pesos de ImageNet sin borrarlos.
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_FASE2, clipnorm=1.0),
                   loss=criterio_loss, metrics=['accuracy'])
    modelo.summary(show_trainable=True)
    hist_f2 = modelo.fit(
        datos_entrenamiento,
        validation_data=dataset_validacion,
        epochs=EPOCAS_FASE2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping(), reducir_lr()],
        verbose=2
    )

    # Las dos fases se unen en una sola curva continua.
    historial = {
        'train_loss': hist_f1.history['loss'] + hist_f2.history['loss'],
        'val_loss':   hist_f1.history['val_loss'] + hist_f2.history['val_loss'],
        'train_acc':  hist_f1.history['accuracy'] + hist_f2.history['accuracy'],
        'val_acc':    hist_f1.history['val_accuracy'] + hist_f2.history['val_accuracy'],
        'epoca_fin_fase1': epoca_fin_fase1
    }

    ruta_modelo = ruta_salida(f"modelo_{NOMBRE_MODELO}_{nombre_experimento}.keras")
    modelo.save(ruta_modelo)
    with open(ruta_salida(f"clases_{NOMBRE_MODELO}_{nombre_experimento}.json"), "w", encoding="utf-8") as archivo:
        json.dump(class_names, archivo, ensure_ascii=False, indent=2)
    print(f"\nModelo guardado en: {ruta_modelo}")
    if EXPORTAR_LITERT:
        exportar_litert(modelo, nombre_experimento)
    print()

    return modelo, historial


modelo_sin_aug, hist_sin_aug = entrenar_modelo(usar_aug=False, nombre_experimento="Sin_Data_Augmentation")
modelo_con_aug, hist_con_aug = entrenar_modelo(usar_aug=True,  nombre_experimento="Con_Data_Augmentation")

# 8. CURVAS DE APRENDIZAJE
def graficar_curvas(historial, titulo_modelo, prefijo_archivo):
    epocas = range(1, len(historial['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epocas, historial['train_acc'], label='Entrenamiento', color='#1f77b4', linewidth=2)
    ax1.plot(epocas, historial['val_acc'], label='Validación', color='#ff7f0e', linewidth=2)
    ax1.set_title(f'Exactitud - {titulo_modelo}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Exactitud')
    ax1.set_ylim([0.0, 1.05])

    ax2.plot(epocas, historial['train_loss'], label='Entrenamiento', color='#d62728', linewidth=2)
    ax2.plot(epocas, historial['val_loss'], label='Validación', color='#2ca02c', linewidth=2)
    ax2.set_title(f'Pérdida - {titulo_modelo}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Entropía cruzada')

    # La línea punteada marca el inicio del ajuste fino.
    for ax in (ax1, ax2):
        ax.axvline(x=historial['epoca_fin_fase1'] + 0.5, color='black', linestyle=':', linewidth=1.8, label='Inicio Fine-Tuning')

    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(f"Curvas de entrenamiento: {titulo_modelo}", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    guardar_figura(fig, f'curvas_{NOMBRE_MODELO}_{prefijo_archivo}.png')


graficar_curvas(hist_sin_aug, "ResNet50 Sin Data Augmentation", "sin_aug")
graficar_curvas(hist_con_aug, "ResNet50 Con Data Augmentation", "con_aug")


def graficar_comparacion(hist_sin, hist_con, titulo_modelo):
    """Validación SIN vs CON aumento en la misma gráfica (entrenamiento más tenue)."""
    fig, ejes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, clave, titulo in [(ejes[0], 'acc', 'Exactitud'), (ejes[1], 'loss', 'Pérdida')]:
        for hist, nombre, color in [(hist_sin, 'SIN aumento', '#d62728'), (hist_con, 'CON aumento', '#2ca02c')]:
            epocas = range(1, len(hist[f'val_{clave}']) + 1)
            ax.plot(epocas, hist[f'val_{clave}'], color=color, linewidth=2, label=f'Validación {nombre}')
            ax.plot(epocas, hist[f'train_{clave}'], color=color, linewidth=1, alpha=0.35,
                    label=f'Entrenamiento {nombre}')
        ax.set_title(f'{titulo} - {titulo_modelo}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Épocas')
        ax.set_ylabel(titulo)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    plt.tight_layout()
    guardar_figura(fig, f'comparacion_{NOMBRE_MODELO}_sin_vs_con.png')


graficar_comparacion(hist_sin_aug, hist_con_aug, "ResNet50")

# 9. EVALUACIÓN EN EL CONJUNTO DE PRUEBA
def evaluar_en_test(modelo, titulo_modelo, prefijo_archivo):
    # El conjunto de prueba se usa una sola vez, aquí, con el modelo ya entrenado.
    y_pred = np.argmax(modelo.predict(dataset_prueba, verbose=0), axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    errores = int(np.sum(y_test != y_pred))

    print("\n" + "=" * 75)
    print(f" RESULTADOS EN PRUEBA: {titulo_modelo.upper()}")
    print("=" * 75)
    print(classification_report(y_test, y_pred, labels=np.arange(num_clases),
                                target_names=class_names, digits=4, zero_division=0))

    # Matriz de confusion
    etiquetas_matriz = [nombre.replace("_", "\n") for nombre in class_names]
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=etiquetas_matriz).plot(
        ax=ax, cmap='Blues', xticks_rotation=30, values_format='d', colorbar=False)
    ax.set_title(f'Matriz de confusión: {titulo_modelo}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Familia predicha')
    ax.set_ylabel('Familia real')
    ax.tick_params(axis='both', labelsize=8)
    fig.tight_layout()
    guardar_figura(fig, f'matriz_confusion_{NOMBRE_MODELO}_{prefijo_archivo}.png')

    # Ejemplos de predicción: primero los errores y se completa con aciertos hasta 9 imágenes.
    # Título verde = acierto, rojo = error.
    orden = np.concatenate([np.where(y_pred != y_test)[0], np.where(y_pred == y_test)[0]])[:9]
    fig = plt.figure(figsize=(12, 8))
    for i, indice in enumerate(orden):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[indice].squeeze(), cmap="gray", vmin=0, vmax=255)
        acierto = y_pred[indice] == y_test[indice]
        plt.title(f"Real: {class_names[y_test[indice]]}\nPred: {class_names[y_pred[indice]]}",
                  color='green' if acierto else 'red', fontsize=9)
        plt.axis("off")
    plt.suptitle(f"Predicciones en prueba: {titulo_modelo}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    guardar_figura(fig, f'predicciones_{NOMBRE_MODELO}_{prefijo_archivo}.png')

    print(f"Exactitud: {acc:.4f} | F1 macro: {f1_macro:.4f} | Errores: {errores}/{len(y_test)}")
    return {'accuracy': acc, 'f1_macro': f1_macro, 'errores': errores, 'total': int(len(y_test))}


metricas_sin_aug = evaluar_en_test(modelo_sin_aug, "ResNet50 Sin Data Augmentation", "sin_aug")
metricas_con_aug = evaluar_en_test(modelo_con_aug, "ResNet50 Con Data Augmentation", "con_aug")

# 10. RESUMEN COMPARATIVO
print("\n" + "#" * 72)
print(f" RESUMEN COMPARATIVO EN PRUEBA: {NOMBRE_MODELO.upper()}")
print("#" * 72)
print(f"{'Métrica':<22} | {'Sin Augmentation':<22} | {'Con Augmentation':<22}")
print("-" * 72)
for etiqueta, clave in [("Accuracy", 'accuracy'), ("F1-Score (Macro)", 'f1_macro')]:
    print(f"{etiqueta:<22} | {metricas_sin_aug[clave]:<22.4f} | {metricas_con_aug[clave]:<22.4f}")
errores_sin = f"{metricas_sin_aug['errores']}/{metricas_sin_aug['total']}"
errores_con = f"{metricas_con_aug['errores']}/{metricas_con_aug['total']}"
print(f"{'Errores / total':<22} | {errores_sin:<22} | {errores_con:<22}")
print("#" * 72)

hora_hermosillo = datetime.datetime.now(zona_hermosillo)
print(f"\nFinalizado en Hermosillo: {hora_hermosillo:%H:%M:%S}")
print(f"Tiempo total de ejecución: {(time.time() - tiempo_inicial) / 60:.2f} minutos")
