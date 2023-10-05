import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Estandaritzar les dades: StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Entrenam una SVM linear (classe SVC)
svc = SVC(C=10000.0, kernel="linear")
hola = svc.fit(X_train, y_train)

# Prediccio
output = svc.predict(X_test)

# Metrica
num_differences = np.sum(np.abs(output-y_test))
estimator = ((len(y_test)-num_differences)/len(y_test))*100
print("SVC has a percentage of success of: "+str(estimator))