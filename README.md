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

Este proyecto implementa una red neuronal basada en [RTDETR](https://docs.ultralytics.com/models/rtdetr/) para detectar somnolencia en conductores de vehículos, considerando condiciones de conducción diurna y nocturna.
Inicialmente orientado a la predicción de landmarks oculares, el enfoque fue adaptado a un modelo de detección de objetos para una mayor robustez en diferentes escenarios.

## 🎯 Objetivo

Desarrollar un modelo capaz de detectar ojos abiertos y cerrados en tiempo real para inferir niveles de somnolencia, contribuyendo a la prevención de accidentes de tráfico.

<details open>
<summary>Entorno de entrenamiento</summary>

Se utilizo la infraestructura [HPC CEDIA](https://cedia.edu.ec/beneficio/supercomputador/) Esta plataforma facilita el acceso a supercomputadoras y clusters de procesamiento optimizados para tareas computacionalmente intensivas como el entrenamiento de modelos de inteligencia artificial, simulaciones científicas, análisis de big data, procesamiento de imágenes y bioinformática.

**Configuración del entorno:**

- `Tipo de Runtime:` Python 3
- `Cores de CPU:` 40 cores
- `Acelerador de hardware:` GPU NVIDIA A100 SMX4
- `Capacidad de la GPU:` Memoria dedicada de 40 GB
- `Memoria RAM:` 32 GB

**Versiones de herramientas y librerías claves:**

- `Python:` 3.8.12
- `PyTorch:` 2.4.1+cu121
- `Ultralytics:` 8.3.58

👁️ Este proyecto utiliza Ultralytics para la detección de objetos. Instala las dependencias ejecutando:

```bash
pip install ultralytics==8.3.58
```
</details>
