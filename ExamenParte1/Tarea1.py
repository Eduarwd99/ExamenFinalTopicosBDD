# Eduardo Arizala Dueñas

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import tree, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Reading the CSV file
df = pd.read_csv("iris_csv.csv")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

iris = load_iris() # Load Data
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target
df['target name'] = df['target'].apply(lambda x: 'sentosa' if x == 0 else ('versicolor' if x == 1 else 'virginica'))

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

#Printing
print(df)

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

n_labels = len(set(df['target']))
labels = ['sentosa', 'versicolor', 'virginica']
print(f'Number of labels: {n_labels}')
print(f"labels: {set(df['target name'])}")

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

print(df.describe())

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

nan_count = df.isna().sum()
print(nan_count )

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

print(df.value_counts("target"))

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

y = df.target
y_names = df["target name"]
X_2d = df.iloc[:, :2]
print(X_2d.head())
print("----------------------------------------------------------")
print(y)
print("----------------------------------------------------------")
print(y_names)

print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")

X_train_2d, X_test_2d, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, shuffle=True, random_state=42)
print(f"{len(X_train_2d)} training examples")
print(f"{len(X_test_2d)} test examples")

plt.figure(2, figsize=(8, 6))
plt.clf()
for label_id, label in enumerate(df):
    X_temp = X_train_2d.loc[y_train == label_id]
    plt.scatter(X_temp.iloc[:, 0], X_temp.iloc[:, 1], cmap=plt.cm.Set1, edgecolor="k", label=label)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xticks()
plt.yticks()
plt.legend()
#plt.show()   #Habilitar para mostrar los grafico de conjunto de entrenamiento (solo usar un plt.show() al final del codigo)

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

svm_model = SVC(gamma=0.1, kernel="rbf", probability=True)
svm_model.fit(X_train_2d, y_train)
y_test_pred_svm = svm_model.predict(X_test_2d)
cm = confusion_matrix(y_test, y_test_pred_svm)
print(cm)

cmd = ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_svm,
cmap=plt.cm.Blues)
ax = cmd.ax_
ax.set_title('Confusion Matrix')
#ax.set_xticklabels('Predict label')
#ax.set_yticklabels('True label')
cmd.display_labels
#plt.show() #Habilitar para mostrar los grafico de matriz de confusion (solo usar un plt.show() al final del codigo)

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

acc_svm = accuracy_score(y_test, y_test_pred_svm)
f1_svm = f1_score(y_test, y_test_pred_svm, average='macro')
print(f"Accuracy: {acc_svm:.2}")
print(f"F1: {f1_svm:.2}")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

dt_model = DecisionTreeClassifier(max_depth=4)
dt_model.fit(X_train_2d, y_train)
y_test_pred_dt = dt_model.predict(X_test_2d)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_2d, y_train)
y_test_pred_knn = knn_model.predict(X_test_2d)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

rf_model = RandomForestClassifier(max_depth=2)
rf_model.fit(X_train_2d, y_train)
y_test_pred_rf = rf_model.predict(X_test_2d)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Plotting decision regions
from itertools import product

f, axarr = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))
for idx, clf, tt in zip(product([0, 1], [0, 1]), 
                        [svm_model, dt_model, knn_model, rf_model],
                        ["SVM", "Decision Tree", "KNN", "Random Forest"]):
    DecisionBoundaryDisplay.from_estimator(
        clf, X_train_2d, alpha=0.4, ax=axarr[idx[0], idx[1]], response_method="predict"
    )
    axarr[idx[0], idx[1]].scatter(X_train_2d.iloc[:, 0], X_train_2d.iloc[:, 1], c=y_train, s=20, edgecolor="k")
    axarr[idx[0], idx[1]].set_title(tt)

#plt.show()

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

print(labels)

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

classification_report_svm = classification_report(y_test, y_test_pred_svm, target_names=labels)
classification_report_dt = classification_report(y_test, y_test_pred_dt, target_names=labels)
classification_report_knn = classification_report(y_test, y_test_pred_knn, target_names=labels)
classification_report_rf = classification_report(y_test, y_test_pred_rf, target_names=labels)

print("SVM")
print(classification_report_svm)
print("\n\nDecision Tree")
print(classification_report_dt)
print("\n\nKNN")
print(classification_report_knn)
print("\n\nRandom Forest")
print(classification_report_rf)

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

y = df.target
y_names = df["target name"]
X = df.iloc[:, :4]
print(X.head())

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
shuffle=True)
print(f"Number of training examples {len(X_train)}")
print(f"Number of test examples {len(X_test)}")

print("----------------------------------------------------------")
print("----------------------------------------------------------")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("----------------------------------------------------------")
print("----------------------------------------------------------")

plt.show()


