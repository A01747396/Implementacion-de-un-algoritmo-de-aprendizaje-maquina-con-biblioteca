# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np

# Backend no-interactivo para guardar figuras sin GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    learning_curve, validation_curve, GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.tree import export_graphviz
import graphviz

# Guardar los resultados en una carpeta
RESULTS = Path("resultados_buenos")
RESULTS.mkdir(exist_ok=True, parents=True)

# Dataset de vino
ds = load_wine()
X, y = ds.data, ds.target
class_names = ds.target_names
feature_names = ds.feature_names

# Dividir el dataset en 60% train, 40% test (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=42
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search para hiperparámetros
param_grid = {
    "max_depth": [3, 4, 5, 6],
    "min_samples_leaf": [2, 3, 4, 6],
    "min_samples_split": [4, 8, 12],
    "max_features": ["sqrt", 0.5, 0.7],
}
base_rf = RandomForestClassifier(
    n_estimators=400, bootstrap=True, random_state=42, n_jobs=-1
)

gs = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
    verbose=0
)
gs.fit(X_train, y_train)

print("\n=== GridSearchCV ===")
print("Mejores params:", gs.best_params_)
print(f"CV best score (accuracy): {gs.best_score_:.4f}")

# Usar el mejor modelo
rf = gs.best_estimator_

# CV con el mejor modelo (para reporte)
cv_acc = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"CV Accuracy (best RF): {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

# Entrenar y evaluar en test
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print("\n=== Test ===")
print(f"Accuracy (test): {test_acc:.4f}")
print("\n=======Classification Report======")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# Matriz de confusión (simple y legible)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de confusión (Wine)")
disp.ax_.figure.tight_layout()
disp.ax_.figure.savefig(RESULTS / "confusion_matrix.png", bbox_inches="tight")
plt.close(disp.ax_.figure)

# Importancias de características (Top 10) - horizontal
importances = rf.feature_importances_
order = np.argsort(importances)[::-1][:10]
fig = plt.figure(figsize=(7, 5), dpi=130)
plt.barh(range(len(order)), importances[order])
plt.yticks(range(len(order)), [feature_names[i] for i in order])
plt.gca().invert_yaxis()  # la más importante arriba
plt.xlabel("Importance")
plt.title("Random Forest - Feature Importances (Top 10)")
fig.tight_layout()
fig.savefig(RESULTS / "feature_importances.png", bbox_inches="tight")
plt.close(fig)

# Learning curve (para diagnosticar overfitting)
train_sizes, train_scores, valid_scores = learning_curve(
    rf, X_train, y_train, cv=cv, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
)
fig = plt.figure(dpi=130)
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train (CV mean)")
plt.plot(train_sizes, valid_scores.mean(axis=1), "o-", label="Validation (CV mean)")
plt.xlabel("Training samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Best Random Forest (Wine)")
plt.legend()
fig.tight_layout()
fig.savefig(RESULTS / "learning_curve.png", bbox_inches="tight")
plt.close(fig)

# Validation curve: max_depth
md_range = [2, 3, 4, 5, 6, 8, 10]
train_vc_md, valid_vc_md = validation_curve(
    RandomForestClassifier(
        n_estimators=rf.n_estimators,
        min_samples_leaf=rf.min_samples_leaf,
        min_samples_split=rf.min_samples_split,
        max_features=rf.max_features,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    X_train, y_train,
    param_name="max_depth", param_range=md_range,
    cv=cv, scoring="accuracy", n_jobs=-1
)
fig = plt.figure(dpi=130)
plt.plot(md_range, train_vc_md.mean(axis=1), "o-", label="Train")
plt.plot(md_range, valid_vc_md.mean(axis=1), "o-", label="Validation")
plt.xlabel("max_depth"); plt.ylabel("Accuracy")
plt.title("Validation Curve - max_depth")
plt.legend()
fig.tight_layout()
fig.savefig(RESULTS / "validation_curve_max_depth.png", bbox_inches="tight")
plt.close(fig)

# Validation curve: min_samples_leaf
leaf_range = [1, 2, 3, 4, 6, 8, 10]
train_vc_leaf, valid_vc_leaf = validation_curve(
    RandomForestClassifier(
        n_estimators=rf.n_estimators,
        max_depth=rf.max_depth,
        min_samples_split=rf.min_samples_split,
        max_features=rf.max_features,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    X_train, y_train,
    param_name="min_samples_leaf", param_range=leaf_range,
    cv=cv, scoring="accuracy", n_jobs=-1
)
fig = plt.figure(dpi=130)
plt.plot(leaf_range, train_vc_leaf.mean(axis=1), "o-", label="Train")
plt.plot(leaf_range, valid_vc_leaf.mean(axis=1), "o-", label="Validation")
plt.xlabel("min_samples_leaf"); plt.ylabel("Accuracy")
plt.title("Validation Curve - min_samples_leaf")
plt.legend()
fig.tight_layout()
fig.savefig(RESULTS / "validation_curve_min_samples_leaf.png", bbox_inches="tight")
plt.close(fig)

# Graficar un árbol con Graphviz (PNG)
tree0 = rf.estimators_[0]
dot = export_graphviz(
    tree0,
    out_file=None,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    impurity=True,
    proportion=True,
    max_depth=3
)
graph = graphviz.Source(dot)
graph.format = "png"
graph.render(str(RESULTS / "rf_tree0_graphviz"), cleanup=True)

print("\nArchivos guardados en:", RESULTS.resolve())
print("- confusion_matrix.png")
print("- feature_importances.png")
print("- learning_curve.png")
print("- validation_curve_max_depth.png")
print("- validation_curve_min_samples_leaf.png")
print("- rf_tree0_graphviz.png")
