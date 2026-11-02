# Clasificación de Familias de Caracoles con Transfer Learning

Clasificación de imágenes SEM (microscopio electrónico de barrido) de caracoles
en **6 familias** usando *transfer learning* con dos arquitecturas preentrenadas
en ImageNet: **VGG16** y **ResNet50**.

- **Demo en vivo:** https://clasificadorcaracolesdani.netlify.app/
- **Artículo en Medium:** https://medium.com/@alvarezdanieleduardo/clasificaci%C3%B3n-de-caracoles-con-transfer-learning-comparativa-entre-resnet50-y-vgg16-f81a8b901820?postPublishedType=repub

Cada arquitectura se entrena dos veces con la misma partición de datos:

- **Sin Data Augmentation**: imágenes originales tal cual.
- **Con Data Augmentation**: rotaciones suaves (±15°), traslaciones (5 %) y ruido
  gaussiano tenue con Albumentations, aplicados solo al conjunto de entrenamiento.

El objetivo es comparar el efecto del aumento de datos en cada arquitectura.

## Dataset

El dataset no está incluido en este repositorio. Se espera un directorio con una
subcarpeta por familia (estructura estándar de `image_dataset_from_directory`):

```
CARACOLES/
├── Cerithiidae/
├── Embrionic_shell/
├── Eulimidae/
├── Litiopidae/
├── Neritidae/
└── Risooinidae/
```

Distribución de clases (2,964 imágenes PNG en escala de grises, 712×423 px):

| Familia          | Imágenes |
|------------------|----------|
| Cerithiidae      | 687      |
| Embrionic_shell  | 466      |
| Eulimidae        | 388      |
| Litiopidae       | 561      |
| Neritidae        | 704      |
| Risooinidae      | 158      |

Las imágenes se reducen a **377×224** conservando la proporción original
(interpolación `area`, sin estirar ni rellenar) y el canal gris se replica a
RGB. La partición es estratificada **70% / 15% / 15%**
(entrenamiento / validación / prueba, semilla fija). Para compensar el
desbalance (Risooinidae es la clase minoritaria) se usan pesos de clase con
`compute_class_weight('balanced')`.

## Estructura del repositorio

```
.
├── caracoles.yml                               # Entorno conda
├── README.md
├── .gitignore
├── VGG16/
│   ├── keras_transfer_learning_vgg16_caracoles.py
│   └── Resultados/                             # Gráficas de la corrida
└── ResNet/
    ├── keras_transfer_learning_resnet50_caracoles.py
    └── Resultados/
```

Cada `Resultados/` contiene las gráficas generadas: distribución de clases,
muestras originales/aumentadas, curvas de aprendizaje, comparativa sin/con
aumento, curvas ROC, matrices de confusión (absoluta y normalizada) y ejemplos
de predicción.

## Metodología

Ambos scripts siguen el mismo flujo:

1. **Carga y análisis** del dataset (gráfica de distribución por familia).
2. **Partición estratificada** 70/15/15 y cálculo de pesos de clase.
3. **Visualización** de muestras originales vs. aumentadas.
4. **Arquitectura:** base preentrenada con `include_top=False` +
   `Rescaling` (normalización dentro del modelo) + `GlobalAveragePooling2D` +
   `Dropout(0.4)` + `Dense(6, softmax)`. La entrada son píxeles 0–255: VGG16
   divide entre 255 y ResNet50 resta 114.8 en los tres canales (misma resta
   para mantener la imagen gris y respetar la escala de sus BatchNormalization).
5. **Entrenamiento en dos fases**:
   - *Fase 1*: solo la cabecera con la base congelada, `Adam(lr=1e-3)`,
     hasta 15 épocas.
   - *Fase 2*: ajuste fino únicamente del último bloque convolucional
     (VGG16: `block5`; ResNet50: `conv5`, con todas las BatchNormalization
     congeladas), `Adam(lr=1e-5)`, hasta 15 épocas.
   - `clipnorm=1.0` en ambas fases; callbacks `EarlyStopping` (restaura los
     mejores pesos) y `ReduceLROnPlateau`.
6. **Evaluación final** sobre el conjunto de prueba (usado una sola vez):
   reporte de clasificación, matriz de confusión y figura de predicciones.

## Requisitos

El entorno se puede reproducir con el archivo conda incluido:

```bash
conda env create -f caracoles.yml
conda activate caracoles
```

Dependencias principales: Python 3.10, TensorFlow/Keras 3, scikit-learn,
matplotlib, numpy, albumentations + opencv. (En el entorno original se usó
`tensorflow-rocm` para GPUs AMD Instinct MI210; en equipos con NVIDIA o CPU
basta instalar `tensorflow` estándar con pip.)

## Uso

Edita la variable `folder_path` en cada script para apuntar al directorio del
dataset y ejecuta:

```bash
cd VGG16
python keras_transfer_learning_vgg16_caracoles.py

cd ../ResNet
python keras_transfer_learning_resnet50_caracoles.py
```

Cada corrida genera en el directorio de trabajo:

- `modelo_<arq>_Sin_Data_Augmentation.keras` y `..._Con_Data_Augmentation.keras`
- `modelo_<arq>_Sin_Data_Augmentation.tflite` y `..._Con_Data_Augmentation.tflite`
  (float16, para la página web; entrada `[1, 224, 377, 3]` con valores 0–255)
- `clases_*.json`
- Gráficas PNG: distribución de clases, muestras originales/aumentadas,
  curvas de aprendizaje, comparativas sin/con aumento, matrices de confusión
  y ejemplos de predicción. (En `Resultados/` del repo hay además curvas ROC
  y matrices normalizadas de una corrida anterior.)

> **Nota:** los archivos `.keras` (modelos entrenados) están excluidos del
> repositorio vía `.gitignore` por el límite de 100 MB de GitHub. Los `.tflite`
> (29–47 MB) se copian a `web/modelos/` para la demo.

## Resultados en prueba (445 imágenes)

| Modelo                 | Accuracy | F1 macro | Errores |
|------------------------|----------|----------|---------|
| VGG16 sin aumento      | 0.9955   | 0.9960   | 2/445   |
| VGG16 con aumento      | 0.9933   | 0.9942   | 3/445   |
| **ResNet50 sin aumento** | **0.9978** | **0.9980** | **1/445** |
| ResNet50 con aumento   | 0.9978   | 0.9980   | 1/445   |

El mejor modelo fue ResNet50 sin aumento, con un único error (una Neritidae
clasificada como Embrionic_shell). El data augmentation no mejoró el desempeño
en este dataset homogéneo.

![Distribución de familias](ResNet/Resultados/distribucion_caracoles_resnet50.png)

## Página web

La demo publicada en Netlify es una página estática que ejecuta los modelos
`.tflite` en el navegador con **LiteRT.js** (WebGPU si está disponible, WASM en
otro caso). Permite subir una imagen, usar la cámara o probar con ejemplos del
dataset. La inferencia es local: ninguna imagen sale del dispositivo.

https://clasificadorcaracolesdani.netlify.app/
