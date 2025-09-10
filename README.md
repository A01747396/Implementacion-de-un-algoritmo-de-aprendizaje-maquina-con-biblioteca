# Clasificación de Vinos con Random Forest

Este repositorio contiene la implementación de un **Random Forest Classifier** aplicado al dataset **Wine** de scikit-learn.  
El proyecto se desarrolló con fines académicos y forma parte de un portafolio de aprendizaje automático supervisado.  

## 📖 Descripción
El modelo clasifica tres variedades de vino italiano a partir de 13 atributos químicos (alcohol, flavonoides, intensidad de color, etc.).  
Se utilizaron técnicas de validación cruzada y optimización de hiperparámetros para garantizar robustez y evitar sobreajuste.  

## ⚙️ Tecnologías utilizadas
- Python 3.12  
- scikit-learn  
- matplotlib  
- numpy  
- graphviz  

Los resultados se guardarán automáticamente en la carpeta creada.

📊 Resultados principales

Accuracy en validación cruzada: 0.9810 ± 0.0233

Accuracy en conjunto de prueba: 0.9722

Métricas balanceadas en las tres clases: precisión, recall y F1 > 0.95

Variables más importantes: proline, alcohol, color_intensity, flavanoids

📈 Visualizaciones incluidas

Matriz de confusión

Curva de aprendizaje

Curvas de validación para max_depth y min_samples_leaf

Importancia de características (Top 10)

Visualización de un árbol individual (Graphviz)
