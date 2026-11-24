""" descente de gradient avec numpy"""
#Chargement des modules/packages
import numpy as np
import matplotlib.pyplot as plt
#Données d'entrainement 
x=np.array([1,2,3,4,5])
y=np.array([3,5,7,9,11])
#Itialisation des paramètres a et b
a=0
b=0
alpha=0.01 # rate learning/taux d'apprentissage
iterations=2000
n=len(x)
#Liste pour conserver l'évolution de coût
couts=[] 

for i in range(iterations):
    #calcul des prédictions
    y_pred=a*x+b
    erreurs=y_pred-y
    #calcul des dérivées partielles
    da=np.sum(erreurs*x)/n
    db=np.sum(erreurs)/n
    #Mise à jours de a et b
    a=a-alpha*da
    b=b-alpha*db
    #Calcul du coût après la mises de a et b
    y_pred=a*x+b
    cout=np.sum((y_pred-y)**2)/(2*n)
    couts.append(cout)
print("Coéfficient a=",a)
print("Coéfficient b=",b)
print("Coût final:",couts[-1])

#Visualisation des données et de droites
y_final=a*x+b
plt.scatter(x,y,label='Les données réelles')
#plt(x,y_final,label='Droite apprise')
plt.xlabel('X')
plt.ylabel("Y")
plt.grid()
plt.legend()
plt.title('Desçente de gradient')
plt.show()