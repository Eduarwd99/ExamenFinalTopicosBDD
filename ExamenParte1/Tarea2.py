# Eduardo Arizala Dueñas

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Generar datos de ejemplo
X, y = make_classification(n_samples=12, n_features=2, n_informative=2, n_redundant=0, random_state=123, n_clusters_per_class=1)

# Crear y entrenar el modelo SVM
svm_model = SVC(gamma=1.1, kernel="rbf", probability=True)
svm_model.fit(X, y)

# Crear una cuadrícula para visualizar las predicciones del modelo
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predecir las clases para cada punto en la cuadrícula
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar los resultados
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Label X')
plt.ylabel('Label Y')
plt.title('SVM')
plt.show()


