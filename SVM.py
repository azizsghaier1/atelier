import numpy as np
import matplotlib.pyplot as plt
import random
def UZAWA(x, y,niter=70000 , pas = 0.0005,eps = 10e-6):
    lamda = np.array([0.1] * x.shape[0]) #-> dimension de lambda=nombre d'eperience
    i = 0
    while(i < niter):
        w = np.zeros(3) # initialisation à zéro
        gather = lamda.copy()
        Margin = [] # intialisation de marge
        L = []
        #Détermination de ω1 et ω2
        w[:2] = np.dot(y * lamda, x)
        #Détermination des points sur les vecteurs supports tels que λ > 0
        for l in range(x.shape[0]):
            if gather[l] > 0:
                Margin.append(x[l])
        #Détermination de ω3
        if Margin != []:
            for m in Margin:
                w[-1] += np.dot(m, w[:2])
            w[-1] /= len(Margin)
        #Calcul des contraintes gi(ω) pour chaque individu
        for j in range(len(x)):
            L.append(contrainte(x, y, w, j))
        g = np.array(L)
        #Calcul de λ selon la méthode d'Uzawa
        lamda += pas * g
        for j in range(lamda.shape[0]):
            if lamda[j] < 0:
                lamda[j] = 0

        if np.linalg.norm(lamda - gather, 2) < eps:
           return w
        i += 1
    return w
  
#Fonction qui représente graphiquement la solution
def visualiser(x, w):
    w = list(w)
    #X est l'axe des abscisses et y est l'axe des ordonnées
    X = x[:, 0]
    y = x[:, 1]
    #Détermination des 3 droites à dessiner : D0, D1 et D-1
    d0 = (- w[0] * X + w[2]) / w[1]
    # droites de marges
    d1 = (- w[0] * X + w[2] + 1) / w[1]
    d_1 = (- w[0] * X + w[2] - 1) / w[1]
    plt.title("Support Vector Machine")
    plt.xlabel('Taille(m)')
    plt.ylabel('Poids(Kg)')
    plt.scatter(X, y, label = 'Individu', color = 'saddlebrown')
    plt.plot(X, d0, label='D0', linestyle='dashed', color = 'gray')
    plt.plot(X, d1, label='D1', color = 'darkblue')
    plt.plot(X, d_1, label='D-1', color = 'lightblue')
    plt.show()

#Génération des données de test
individu = np.array([[1.5, 72], [1.6, 77], [1.65, 80], [1.75, 92], [1.6, 60], [1.83, 69], [1.7, 65], [1.8, 72]])
classification = np.array([1, 1, 1, 1, -1, -1, -1, -1])
for i in range(16):
    data = [random.uniform(1.5, 2.0), random.uniform(50, 100)]
    individu = np.concatenate((individu, np.array([data])))
    imc = data[1] / data[0] ** 2
    if imc < 24.9:
        classification = np.concatenate((classification, np.array([-1])))
    else:
        classification = np.concatenate((classification, np.array([1])))

#Test de la fonction qui calcule les paramètres ω et λ
W = UZAWA(individu, classification)
#Test de la fonction qui génère la représentation graphique
visualiser(individu, W)
