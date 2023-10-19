from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# En realitat ja no necessitem canviar les etiquetes Scikit ho fa per nosaltres

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Els dos algorismes es beneficien d'estandaritzar les dades

scaler = MinMaxScaler() #StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

def creaSVC(kernel,string):
    svm = SVC(C=1.0, kernel=kernel, random_state=33)
    svm.fit(X_transformed, y_train)
    y_meu = svm.predict(X_test_transformed)
    accuracy=precision_score(y_test, y_meu)
    print("La precisió del kernel "+str(string)+" és: "+str(accuracy))

#Custom linear kernel
def kernel_lineal(x1, x2):
    return x1.dot(x2.T)

#Gaussian kernel



def gaussian_kernel(x1, x2, sigma=1):
    gamma = 1/(2*(sigma**2))
    dist = distance_matrix(x1, x2)
    return np.exp(-gamma * (dist**2))

def polynomic_kernel(x, y, gamma=10, degree=3):

    return ((gamma * np.dot(x, y.T)) + 0) ** degree

noms    = ['custom_linear_kernel', 'builtin scikit linear ', 'gaussia',
           'scikit gaussian', 'custom polynomial', 'scikit polynomial']
kernels = [kernel_lineal, 'linear', gaussian_kernel, 'rbf', polynomic_kernel, 'poly']

for i in range(6):
    creaSVC(kernels[i], noms[i])

plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_train)

plt.legend()
plt.show()