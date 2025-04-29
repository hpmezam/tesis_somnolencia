<h1 align='center'>Implementación de una red neuronal Transformer para detección de somnolencia en conductores de vehículos durante la conducción diurna y nocturna</h1>
<h3 align='center'>Software Engineer | AI, ML, DL & Computer Vision</h3>
<p align="center">
  <!-- Lenguajes y Frameworks -->
  <img src="https://img.shields.io/badge/Python-3.8.12-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <!-- Machine Learning -->
  <img src="https://img.shields.io/badge/PyTorch-2.4.1+cu121-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Ultralytics-8.3.58-00FFFF?style=for-the-badge&logo=yolo&logoColor=black">
  <img src="https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <!-- Hardware y Deployment -->
  <img src="https://img.shields.io/badge/Raspberry_Pi-A22846?style=for-the-badge&logo=raspberry-pi&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
</p>

## 📖 Descripción

El presente trabajo propone un modelo basado en la arquitectura Transformers [RTDETR](https://docs.ultralytics.com/models/rtdetr/) para la detección de somnolencia, centrándose en la identificación de la zona de los ojos y la evaluación de su estado (despierto o dormido). El modelo será implementado en un sistema embebido con una Raspberry Pi 3 Modelo B+, diseñado para su integración en vehículos y el monitoreo en tiempo real del conductor.

## 🎯 Objetivo

Desarrollar un sistema embebido mediante el uso de visión artificial para detección y alerta de somnolencia en conductores durante la conducción diurna y nocturna.


## 🛠️ Tecnologías Utilizadas

### Entorno de entrenamiento

Se utilizo la infraestructura [HPC CEDIA](https://cedia.edu.ec/beneficio/supercomputador/). esta plataforma facilita el acceso a supercomputadoras y clusters de procesamiento optimizados para tareas computacionalmente intensivas como el entrenamiento de modelos de inteligencia artificial, simulaciones científicas, análisis de big data, procesamiento de imágenes y bioinformática.

<details open>
<summary>Configuración del entorno:</summary>
  
- `Tipo de Runtime:` Python 3
- `Cores de CPU:` 40 cores
- `Acelerador de hardware:` GPU NVIDIA A100 SMX4
- `Capacidad de la GPU:` Memoria dedicada de 40 GB
- `Memoria RAM:` 32 GB
</details>

<details open>
<summary>Versiones de herramientas y librerías claves:</summary>

- `Python:` 3.8.12
- `PyTorch:` 2.4.1+cu121
- `Ultralytics:` 8.3.58
- `OpenCV`
- `NumPy`
- `Matplotlib`

👁️ Este proyecto utiliza Ultralytics para la detección de objetos. Instala las dependencias ejecutando:

```bash
pip install ultralytics==8.3.58
```
</details>

## ⚙️ Entrenamiento

El modelo fue entrenado con un dataset propio de imágenes 128x128 pixeles.

👁️ Se definieron las clases de interés, restringiendo la detección a dos categorías específicas: 

- **Clase 0:** Ojos abiertos (open eye).
- **Clase 1:** Ojos cerrados (closed eye).


```bash
# Importar el modelo
from ultralytics import RTDETR
# Cargar el modelo
model = RTDETR('rtdetr-l.pt')
```

<details open>
<summary>Configuración de Hiperparámetros Personalizada:</summary>

```bash
params = {
  'epochs': 50,
  'imgsz': 128,
  'patience': 5,
  'batch': 8,
  'project': 'Somnolencia',
  'name': 'RTDETR_balanceado2',
  'val': True,
  'plots': True,
  'verbose': True

  # Hiperparámetros
  'optimizer': 'AdamW',
  'cos_lr': True,
  'dropout': 0.5,
  'lr0': 0.0005,
  'weight_decay': 0.005,
  'momentum': 0.95,
  'warmup_epochs': 3.0,
  'warmup_momentum': 0.8,
  'warmup_bias_lr': 0.00
}
```
</details>
<details open>
<summary>Ejecución de entrenamiento:</summary>

```bash
results = model.train(data = 'Datasets/balanced_eye_dataset/data.yaml', classes = [0, 1], **params)
```
</details>

## 📈 Resultados

<details open>
<summary>Dataset General</summary>
  
| Dataset | Loss | Exactitud (%) | Precisión (%) | Recall (%) | F1 Score (%) | Tiempo de entrenamiento |
| ------ | ---- | ------------- | ------------- | ---------- | ------------ | --------- |
| Balanceado | 0.43 | 95.21 | 95.47 | 95.75 | 95.61 | 4h 28m 21s |
| Desbalanceado | 0.41 | 92.62 | 94.60 | 94.41 | 94.50 | 4h 41m 22s |
</details>

<details open>
<summary>Dataset con accesorios faciales</summary>
  
| Dataset Balanceado| Exactitud (%) | Precisión (%) | Recall (%) | F1 Score (%) |
| ------ | ------------- | ------------- | ---------- | ------------ |
| Bareface | 93.0297 | 94.3375 | 94.3333 | 94.3354 |
| Glasses | 92.7173 | 94.5817 | 94.6250 | 94.6034 |
| Sunglasses | 93.4208 | 94.3247 | 94.0809 | 94.2026 |

En comparacion de otros modelos, es de destacar que tiene una buena generalizacion en cuanto a Bareface y glasses pero destaco en sunglasees ya que los otros modelos esos tiene bien bajo quizas debido a la interferencia entre la luna o por ser un mismo objeto

</details>
<details open>
<summary>Dataset en condiciones de luz</summary>
  
| Dataset Balanceado| Exactitud (%) | Precisión (%) | Recall (%) | F1 Score (%) |
| ------ | ------------- | ------------- | ---------- | ------------ |
| Día | 89.4921 | 91.3589 | 91.2271 | 91.293 |
| Noche | 91.7840 | 92.5682 | 92.5 | 92.5342 |
</details>

El modelo [RTDETR](https://docs.ultralytics.com/models/rtdetr/) en su version `rtdetr-l` fue comparado con artículos relacionados con **clasificación**, aunque su arquitectura está enfocada en [detección de objetos](https://www.ibm.com/mx-es/think/topics/object-detection). A pesar de esta diferencia, los principales parámetros de evaluación siguen siendo comparables, ya que incluyen **exactitud, precisión, recall y F1 Score**, métricas ampliamente utilizadas en ambos enfoques. En comparación con otros modelos, RT-DETR demostró una *buena capacidad de generalización* en condiciones sin accesorios (**bareface**) y con gafas (**glasses**), alcanzando precisiones superiores al 94%. No obstante, su desempeño con gafas de sol (**sunglasses**) fue particularmente destacable, superando ampliamente a otros métodos que suelen tener dificultades en esta categoría debido a la interferencia causada por los reflejos oscuros o la ambigüedad visual entre los ojos cerrados y los propios lentes. En esta condición, el modelo obtuvo los siguientes resultados:

- `Exactitud:` 93.4208
- `Precisión:` 94.3247
- `Recall:` 94.0809
- `F1 Score:` 94.2026

## 📚 Referencias

- Vaswani et al., "Attention is All You Need"
- Chen et al., "RT-DETR: Real-Time Detection Transformer"

## ✨ Autor

Henry Meza

<br>
<div align="center">
  <a href="https://github.com/hpmezam">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub">
  </a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/in/hpmezam/">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn">
  </a>
</div>
