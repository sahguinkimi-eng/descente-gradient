""" Desçeente de gradient en utilisant le scikit-learn"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor

# données 
x=np.array([1,2,3,4,5]).reshape(-1,1)
y=np.array([3,5,7,9,11])
#modele
modele=make_pipeline(StandardScaler(),
                     SGDRegressor(loss='squared_error',
                                  max_iter=5000,
                                  tol=1e-7,
                                  learning_rate='constant',
                                  eta0=0.01,
                                  random_state=42))

# entrainement
modele.fit(x,y)
#prédiction
prediction=modele.predict([[6]])
print("Prédiction pour x=6",prediction)