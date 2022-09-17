import numpy as np
import matplotlib.pyplot as plt
def generer_la_matrice(N:int , L:float):
    h=L/N
    A=np.zeros((N-1,N-1))
    A[0,0]=2
    A[1,0]=-1
    A[N-2,N-2]=2
    A[N-3,N-2]=-1
    for i in range(1,N-2):
        A[i,i]=2
        A[i-1,i]=-1
        A[i+1,i]=-1
    return A*(1/h**2)
def generer_b(T0,TL,g:list,N):
    b=np.zeros(N-1)
    b[0]=g[0]+T0
    b[N-2]=g[N-2]+TL
    for i in range(1,N-2) :
        b[i]=g[i]
    return b

#Après la génération de A et b le problème qui se pose c'est de resoudre l'équatopn AX=b
#Pour cela on va résoudre le problème de minimisation de la fonction objective 1/2*<AX,X>-<b,X> par la méthode de gradient descendant
# def la_fonction_objective(A,)
#je vais tout long ce code effectuer trois approche pour résoudre ce problème
#premier Approche: Gradient descendent à pas fix alpha (notons que 0<alpha<2/lambda max ) de préference que alpha=2/(lambda min + lambda max )
def gradient_descendant(T0,TL,g,N,L,X0:list,alpha,n_iter,epsilon) :
    A=generer_la_matrice(N,L)
    b=generer_b(T0,TL,g,N)
    X0=np.array(X0)
    # print('A={} , b={})'.format(A,b))
    f=[1/2*np.dot(np.dot(A,X0),X0)-np.dot(b,X0)]
    for k in range(n_iter) :
        d=-(np.dot(A,X0)-np.array(b))
        X0=X0+alpha*d
        f.append(1/2*np.dot(np.dot(A,X0),X0)-np.dot(b,X0))
        # print('X0=',X0)
        # print('norm=',np.linalg.norm(np.dot(A,X0)-b))
        if np.linalg.norm(np.dot(A,X0)-b)<epsilon :
            break
    # print(f)
    plt.title('le processus de minimisation grad desc')
    plt.xlabel('itérations')
    plt.ylabel('fonction objective')
    plt.plot(f)
    plt.show()
    return 'la solution de gradient_descendant est ',X0
#deuxième approche: Gradient descendent à pas optimal
def gradient_descendant_opt(T0,TL,g,N,L,X0:list,n_iter,epsilon) :
    A=generer_la_matrice(N,L)
    # print(A*(L**2/N**2))
    b=generer_b(T0,TL,g,N)
    X0=np.array(X0)
    f=[1/2*np.dot(np.dot(A,X0),X0)-np.dot(b,X0)]
    # print('A={} , b={})'.format(A,b))
    for i in range(n_iter):
        d=-(np.dot(A,X0)-np.array(b))
        alpha=(np.dot(d,d))/np.dot(np.dot(A,d),d)
        # print('alpha=',alpha)
        X0=X0+alpha*d
        plt.title('le processus de minimisation grad opt')
        plt.xlabel('itérations')
        plt.ylabel('fonction objective')
        f.append( ((1 / 2) * np.dot(np.dot(A, X0), X0) - np.dot(b, X0)) )
        # print('X0=',X0)
        if np.linalg.norm(np.dot(A,X0)-b)<epsilon :
            break
    # print('f=',f)
    plt.plot(f)
    plt.show()
    return 'la solution de gradient_descendant_opt est ',X0
def gradient_descendant_conj(T0,TL,g,N,L,X0:list,n_iter,epsilon) :
    A=generer_la_matrice(N,L)
    b=generer_b(T0,TL,g,N)
    X0=np.array(X0)
    f=[1/2*np.dot(np.dot(A,X0),X0)-np.dot(b,X0)]
    d = -(np.dot(A, X0) - np.array(b))
    alpha = (np.dot(d, d)) / np.dot(np.dot(A, d), d)
    # print('A={} , b={})'.format(A,b))
    for k in range(n_iter) :
        g=(np.dot(A, X0) - np.array(b))
        beta=(np.dot(np.dot(A,g),g)) / np.dot(np.dot(A, d), d)
        d=-g+beta*d
        alpha=-(np.dot(d,g)/np.dot(np.dot(A,d),d))
        X0=X0+alpha*d
        # print(X0)
        f.append( ((1 / 2) * np.dot(np.dot(A, X0), X0) - np.dot(b, X0)) )
        # print('X0=',X0)
        if np.linalg.norm(np.dot(A,X0)-b)<epsilon :
            break
    plt.title('le processus de minimisation grad conj')
    plt.xlabel('itérations')
    plt.ylabel('fonction objective')
    # print('f=',f)
    plt.plot(f)
    plt.show()
    return 'la solution de gradient descendant conjugué est ',X0
alpha=0.01
print(gradient_descendant(T0=10,TL=10,g=[10,20,10],N=4,L=5,X0=[50,250,300],alpha=0.75,n_iter=30,epsilon=0.00001))
print(gradient_descendant_opt(T0=10,TL=10,g=[10,20,10],N=4,L=5,X0=[50,250,300],n_iter=10,epsilon=0.00001))
print(gradient_descendant_conj(T0=10,TL=10,g=[10,20,10],N=4,L=5,X0=[50,250,300],n_iter=10,epsilon=0.00001))
def best_step_grad(T0,TL,g,N,L,X0:list,n_iter,epsilon) :
    A = generer_la_matrice(N, L)
    b = generer_b(T0, TL, g, N)
    X0 = np.array(X0)
    # print('A={} , b={})'.format(A,b))
    alpha=np.linspace(0,1.5,200)
    niter=list()
    for step in alpha:
        X=X0
        for k in range(n_iter):
            d = -(np.dot(A, X) - np.array(b))
            X = X + step * d
            if np.linalg.norm(np.dot(A, X) - b) < epsilon:
                break
        niter.append(k)
    plt.title('trouver le meilleur pas')
    plt.xlabel('pas')
    plt.ylabel("nombre d'itération")
    plt.plot(alpha,niter)
    plt.show()
best_step_grad(T0=10,TL=10,g=[10,20,10],N=4,L=5,X0=[50,250,300],n_iter=50,epsilon=100)
