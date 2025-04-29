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

El modelo fue entrenado con un dataset propio de imágenes 128x128 px, etiquetadas para **detección de ojos abiertos y cerrados**.

```bash
# Importar el modelo
from ultralytics import RTDETR
# Cargar el modelo
model = RTDETR('rtdetr-l.pt')
```

Configuración de Hiperparámetros Personalizada.

```bash
params = {
  'epochs': 50
}
```

Ejecución de entrenamiento.

```bash
results = model.train(data = 'Datasets/balanced_eye_dataset/data.yaml', classes = [0, 1], **params)
```

## 📈 Resultados

- **Precisión:** 
