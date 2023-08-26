# Eduardo Arizala Dueñas

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Supongamos que tienes etiquetas reales y predichas del conjunto de prueba
etiquetas_reales = np.array([0, 10, 2, 3, 9, 5, 6, 8, 9, 6, 10, 0, 1, 2, 3, 4, 5, 1, 7, 8])
etiquetas_predichas = np.array([0, 1, 2, 10, 5, 2, 6, 6, 8, 9, 10, 1, 1, 2, 10, 4, 9, 6, 6, 8])

# Calcular la matriz de confusión
cm = confusion_matrix(etiquetas_reales, etiquetas_predichas, labels=range(11))

# Etiquetas de las clases
labels = [str(i) for i in range(11)]

print("Matriz de Confusión:")
print(cm)

# Crear la matriz de confusión visual
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cmd.plot(cmap=plt.cm.Blues)

# Mostrar la matriz de confusión
plt.show()


