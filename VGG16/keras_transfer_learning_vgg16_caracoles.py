# -*- coding: utf-8 -*-
"""Clasificación de familias de caracoles con VGG16 (transfer learning).

Se entrena el mismo modelo dos veces: una con las imágenes originales y otra con
aumentación de datos. Ambas usan la misma partición; la prueba se reserva para
la evaluación final.
"""

import os
import json
import time
import datetime
import zoneinfo

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    roc_curve,
    roc_auc_score
)

# 1. CONFIGURACIÓN
# Semilla fija para que la partición y la inicialización se repitan igual en cada corrida.
SEED = 42
tf.keras.utils.set_random_seed(SEED)

NOMBRE_MODELO = "vgg16"
folder_path = '/lustre/home/estudiante_02/TransferLearning/CARACOLES'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Topes de épocas por fase; EarlyStopping suele detener antes.
EPOCAS_FASE1 = 15   # cabecera con la base congelada
EPOCAS_FASE2 = 30   # ajuste fino de las capas profundas

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
dataset_completo = keras.utils.image_dataset_from_directory(
    folder_path,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=None,
    shuffle=False
)

class_names = dataset_completo.class_names
num_clases = len(class_names)

# Paso todo a arreglos de numpy para poder partir por índices con scikit-learn.
imagenes_totales = np.empty((len(dataset_completo), *IMG_SIZE, 3), dtype=np.float32)
etiquetas_totales = np.empty(len(dataset_completo), dtype=np.int32)
for i, (img, label) in enumerate(dataset_completo):
    imagenes_totales[i] = img.numpy()
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
plt.savefig(f'distribucion_caracoles_{NOMBRE_MODELO}.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. PARTICIÓN ESTRATIFICADA (60/20/20) Y PESOS DE CLASE
minimo_por_clase = int(conteos_por_clase.min())
if minimo_por_clase < 5:
    clase_pequena = class_names[int(np.argmin(conteos_por_clase))]
    raise ValueError(
        f"La clase {clase_pequena!r} tiene {minimo_por_clase} imágenes. "
        "Se requieren al menos cinco para hacer las dos divisiones estratificadas."
    )

indices = np.arange(len(etiquetas_totales))

# Primero 60% entrenamiento y 40% temporal; luego el temporal se parte en 20/20.
# stratify conserva la proporción de cada familia en las tres particiones.
train_idx, temp_idx, y_train, y_temp = train_test_split(
    indices, etiquetas_totales, test_size=0.40, stratify=etiquetas_totales, random_state=SEED
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

# 4. PIPELINES tf.data
# shuffle mezcla el orden en cada época; prefetch prepara el siguiente lote mientras la GPU entrena.
dataset_train = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=len(x_train), seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
# Validación y prueba no se mezclan ni se aumentan: se evalúan tal cual.
dataset_validacion = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
dataset_prueba     = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def capa_aumentacion():
   return keras.Sequential([
        # Rotación libre (360°): orientación arbitraria del espécimen en la platina.
        layers.RandomRotation(1.0, fill_mode="reflect", seed=SEED),
        # Zoom suave: simula pequeñas diferencias de magnificación entre capturas.
        layers.RandomZoom(0.10, fill_mode="reflect", seed=SEED),
        # Traslación mínima: seguro por si alguna captura queda un poco descentrada.
        layers.RandomTranslation(0.02, 0.02, fill_mode="reflect", seed=SEED),
        # Contraste: variaciones del detector de electrones y ajuste del operador.
        layers.RandomContrast(0.15, seed=SEED),
        # Brillo: acumulación de carga y cambios de kV entre capturas.
        layers.RandomBrightness(0.05, value_range=(0.0, 255.0), seed=SEED),
    ], name="aumentacion")

# 5. VISUALIZACIÓN: IMÁGENES ORIGINALES VS. AUMENTADAS
muestra_imgs = x_train[:9]
muestra_lbls = y_train[:9]
imgs_aumentadas = capa_aumentacion()(muestra_imgs, training=True).numpy()

for imgs, etiqueta, archivo in [
    (muestra_imgs, "Original", "puras"),
    (imgs_aumentadas, "Aumentada", "aumentadas"),
]:
    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[i].astype("uint8"))
        plt.title(f"{etiqueta}: {class_names[muestra_lbls[i]]}", fontsize=10, fontweight='bold')
        plt.axis("off")
    plt.suptitle(f"Muestras de entrenamiento ({etiqueta.lower()}s)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{NOMBRE_MODELO}_muestra_imagenes_{archivo}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. ARQUITECTURA: VGG16 PREENTRENADA + CABECERA
def construir_modelo(usar_aug):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    # Congelada al inicio: en la fase 1 solo aprende la cabecera nueva.
    base_model.trainable = False

    entradas = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = entradas
    # La aumentación vive dentro del modelo y solo actúa durante el entrenamiento.
    if usar_aug:
        x = capa_aumentacion()(x)
    # preprocess_input deja los píxeles como los vio VGG16 al entrenarse en ImageNet.
    x = layers.Lambda(preprocess_input, name="preprocess_vgg16")(x)
    # VGG16 no tiene BatchNormalization, así que no hace falta forzar modo inferencia como en ResNet50.
    x = base_model(x)

    # Global average pooling resume cada mapa en un número: evita una capa densa enorme.
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    # Dropout apaga el 40% de las neuronas al azar en cada paso para que no se memorice el entrenamiento.
    x = layers.Dropout(0.4, seed=SEED, name="dropout")(x)
    # softmax convierte las salidas en una probabilidad por familia que suma 1.
    salidas = layers.Dense(num_clases, activation="softmax", name="clasificador_familias")(x)

    modelo = keras.Model(entradas, salidas, name=f"VGG16_{'con' if usar_aug else 'sin'}_aug")
    return modelo, base_model


def descongelar_capas_profundas(base_model):
    """Libera block4 y block5; block1-block3 quedan congelados.

    Los primeros bloques detectan bordes y texturas genéricas que sirven para
    cualquier imagen; los últimos son los que conviene adaptar a las conchas.
    """
    base_model.trainable = True
    for layer in base_model.layers:
        es_inicial = layer.name.startswith(("block1_", "block2_", "block3_"))
        layer.trainable = not es_inicial
    return sum(layer.trainable for layer in base_model.layers)

# 7. ENTRENAMIENTO EN DOS FASES
def callbacks_entrenamiento(paciencia_es, paciencia_lr, min_lr):
    return [
        # Si la pérdida de validación no mejora en N épocas, se detiene y
        # se recuperan los pesos de la mejor época (evita seguir sobreajustando).
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=paciencia_es, restore_best_weights=True, verbose=1),
        # Si se estanca, baja la tasa de aprendizaje a la mitad para afinar.
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=paciencia_lr, min_lr=min_lr, verbose=1),
    ]


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
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=criterio_loss, metrics=['accuracy'])
    modelo.summary(show_trainable=True)
    hist_f1 = modelo.fit(
        dataset_train,
        validation_data=dataset_validacion,
        epochs=EPOCAS_FASE1,
        class_weight=class_weight_dict,
        callbacks=callbacks_entrenamiento(paciencia_es=5, paciencia_lr=3, min_lr=1e-6),
        verbose=2
    )
    # Se guarda en qué época terminó la fase 1 para marcarla en las gráficas.
    epoca_fin_fase1 = len(hist_f1.history['loss'])

    # Fase 2: con la cabecera ya estable, se dejan ajustar los bloques profundos
    # (los más específicos) para que se adapten a las conchas.
    print("\nFase 2: ajuste fino de las capas profundas")
    capas_entrenables = descongelar_capas_profundas(base_model)
    print(f"Capas entrenables de VGG16: {capas_entrenables}")
    # Recompilar es obligatorio tras cambiar qué capas son entrenables. LR muy baja
    # para ajustar los pesos de ImageNet sin borrarlos.
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-6), loss=criterio_loss, metrics=['accuracy'])
    modelo.summary(show_trainable=True)
    hist_f2 = modelo.fit(
        dataset_train,
        validation_data=dataset_validacion,
        epochs=EPOCAS_FASE2,
        class_weight=class_weight_dict,
        callbacks=callbacks_entrenamiento(paciencia_es=8, paciencia_lr=4, min_lr=1e-7),
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

    ruta_modelo = f"modelo_{NOMBRE_MODELO}_{nombre_experimento}.keras"
    modelo.save(ruta_modelo)
    with open(f"clases_{NOMBRE_MODELO}_{nombre_experimento}.json", "w", encoding="utf-8") as archivo:
        json.dump(class_names, archivo, ensure_ascii=False, indent=2)
    print(f"\nModelo guardado en: {ruta_modelo}\n")

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
    plt.savefig(f'curvas_{NOMBRE_MODELO}_{prefijo_archivo}.png', dpi=300, bbox_inches='tight')
    plt.close()


graficar_curvas(hist_sin_aug, "VGG16 Sin Data Augmentation", "sin_aug")
graficar_curvas(hist_con_aug, "VGG16 Con Data Augmentation", "con_aug")

# 9. EVALUACIÓN EN EL CONJUNTO DE PRUEBA
def intervalo_wilson(aciertos, total, z=1.96):
    """Intervalo de confianza al 95% para una proporción (exactitud).

    Con 593 imágenes de prueba cada error mueve la exactitud 0.17%; el intervalo
    indica qué tan precisa es la cifra y si dos modelos realmente difieren.
    """
    p = aciertos / total
    centro = (p + z**2 / (2 * total)) / (1 + z**2 / total)
    margen = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return centro - margen, centro + margen


def evaluar_en_test(modelo, titulo_modelo, prefijo_archivo):
    # El conjunto de prueba se usa una sola vez, aquí, con el modelo ya entrenado.
    y_pred_probs = modelo.predict(dataset_prueba, verbose=0)
    y_real = y_test
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_real, y_pred)
    f1_macro = f1_score(y_real, y_pred, average='macro')
    f1_weighted = f1_score(y_real, y_pred, average='weighted')
    errores = int(np.sum(y_real != y_pred))
    ic_bajo, ic_alto = intervalo_wilson(len(y_real) - errores, len(y_real))

    print("\n" + "=" * 75)
    print(f" RESULTADOS EN PRUEBA: {titulo_modelo.upper()}")
    print("=" * 75)
    print(classification_report(y_real, y_pred, labels=np.arange(num_clases),
                                target_names=class_names, digits=4, zero_division=0))

    # Curvas ROC uno-contra-resto con las probabilidades de cada familia.
    y_real_bin = np.eye(num_clases)[y_real]
    auc_por_clase = roc_auc_score(y_real_bin, y_pred_probs, average=None)
    auc_macro = roc_auc_score(y_real_bin, y_pred_probs, average='macro')

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, nombre_clase in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_real_bin[:, i], y_pred_probs[:, i])
        ax.plot(fpr, tpr, linewidth=1.7, label=f'{nombre_clase} (AUC={auc_por_clase[i]:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Azar')
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Tasa de falsos positivos',
           ylabel='Tasa de verdaderos positivos',
           title=f'Curvas ROC: {titulo_modelo} (AUC macro={auc_macro:.3f})')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'curva_roc_{NOMBRE_MODELO}_{prefijo_archivo}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Matrices de confusión: conteos y porcentaje dentro de cada familia real.
    etiquetas_matriz = [nombre.replace("_", "\n") for nombre in class_names]
    for normalizar, cmap, formato, sufijo in [(None, 'Blues', 'd', ''), ('true', 'Greens', '.0%', '_normalizada')]:
        cm = confusion_matrix(y_real, y_pred, normalize=normalizar)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(cm, display_labels=etiquetas_matriz).plot(
            ax=ax, cmap=cmap, xticks_rotation=30, values_format=formato, colorbar=False)
        ax.set_title(f'Matriz de confusión{sufijo.replace("_", " ")}: {titulo_modelo}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Familia predicha')
        ax.set_ylabel('Familia real')
        ax.tick_params(axis='both', labelsize=8)
        fig.tight_layout()
        fig.savefig(f'matriz_confusion{sufijo}_{NOMBRE_MODELO}_{prefijo_archivo}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Exactitud: {acc:.4f} (IC 95%: {ic_bajo:.4f}-{ic_alto:.4f}) | "
          f"F1 macro: {f1_macro:.4f} | Errores: {errores}/{len(y_real)} | AUC macro: {auc_macro:.4f}")

    metricas = {'accuracy': acc, 'ic95_bajo': ic_bajo, 'ic95_alto': ic_alto, 'f1_macro': f1_macro,
                'f1_weighted': f1_weighted, 'auc_macro': auc_macro, 'errores': errores, 'total': int(len(y_real))}
    with open(f"metricas_{NOMBRE_MODELO}_{prefijo_archivo}.json", "w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, indent=2)
    return metricas


metricas_sin_aug = evaluar_en_test(modelo_sin_aug, "VGG16 Sin Data Augmentation", "sin_aug")
metricas_con_aug = evaluar_en_test(modelo_con_aug, "VGG16 Con Data Augmentation", "con_aug")

print("\n" + "#" * 72)
print(f" RESUMEN COMPARATIVO EN PRUEBA: {NOMBRE_MODELO.upper()}")
print("#" * 72)
print(f"{'Métrica':<22} | {'Sin Augmentation':<22} | {'Con Augmentation':<22}")
print("-" * 72)
for etiqueta, clave in [("Accuracy", 'accuracy'), ("F1-Score (Macro)", 'f1_macro'),
                        ("F1-Score (Weighted)", 'f1_weighted'), ("AUC (Macro)", 'auc_macro')]:
    print(f"{etiqueta:<22} | {metricas_sin_aug[clave]:<22.4f} | {metricas_con_aug[clave]:<22.4f}")
errores_sin = f"{metricas_sin_aug['errores']}/{metricas_sin_aug['total']}"
errores_con = f"{metricas_con_aug['errores']}/{metricas_con_aug['total']}"
ic_sin = f"{metricas_sin_aug['ic95_bajo']:.4f}-{metricas_sin_aug['ic95_alto']:.4f}"
ic_con = f"{metricas_con_aug['ic95_bajo']:.4f}-{metricas_con_aug['ic95_alto']:.4f}"
print(f"{'Errores / total':<22} | {errores_sin:<22} | {errores_con:<22}")
print(f"{'IC 95% exactitud':<22} | {ic_sin:<22} | {ic_con:<22}")
print("#" * 72)
print("Si los intervalos se solapan, la diferencia entre variantes no es concluyente.")

hora_hermosillo = datetime.datetime.now(zona_hermosillo)
print(f"\nFinalizado en Hermosillo: {hora_hermosillo:%H:%M:%S}")
print(f"Tiempo total de ejecución: {(time.time() - tiempo_inicial) / 60:.2f} minutos")
