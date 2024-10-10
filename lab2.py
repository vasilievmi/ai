1 task
from sklearn.neural_network import MLPClassifier
X=[[0,0],[1,1]]
y=[0,1]
clf=MLPClassifier(solver='lbfgs', alpha=1e5, hidden_layer_sizes=(5,2), random_state=1)
clf.fit(X,y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2),
random_state=1, solver='lbfgs')
clf.predict([[2,2], [-1,-2]])
array([1, 0])
[coef.shape for coef in clf.coefs_]
[(2, 5), (5, 2), (2, 1)]
clf.predict_proba([[2,2],[1,2]])
array([[1.96718015e-04, 9.99803282e-01],
 [1.96718015e-04, 9.99803282e-01]])
#
2 task 
X=[[0,0],[0,1],[1,0],[1,1]]
y=[0,0,0,1]
clf_and=MLPClassifier(solver='lbfgs', alpha=1e5, hidden_layer_sizes=(5,2), random_state=1)
clf_and.fit(X,y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2),
random_state=1, solver='lbfgs')
clf_and.predict([[2,2], [-1,-2], [0,0.9], [0.9,0.9]])
array([1, 0, 0, 1])
#
3 task 
X=[[0,0],[0,1],[1,0],[1,1]]
y=[0,1,1,1]
clf_or=MLPClassifier(solver='lbfgs', alpha=1e5, hidden_layer_sizes=(5,2), random_state=1)
clf_or.fit(X,y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2),
random_state=1, solver='lbfgs')
clf_or.predict([[2,2], [-1,-2], [0,0.9], [0.9,0.9]])
array([1, 0, 1, 1])
#
4 task 
X=[[0,0],[0,1],[1,0],[1,1]]
y=[0,1,1,0]
clf_xor=MLPClassifier(solver='lbfgs', alpha=1e5, hidden_layer_sizes=(5,2), random_state=1)
clf_xor.fit(X,y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2),
random_state=1, solver='lbfgs')
clf_xor.predict([[2,2], [-1,-2], [0,0.5], [0.9,0.9]])
array([0, 0, 0, 1])
import numpy as np
class AdalineGD(object):
 def __init__(self, eta=0.01, n_iter=50, random_state=1):
 self.eta = eta
 self.n_iter = n_iter
 self.random_state = random_state
 self.w_ = None // весовой вектор
 self.w0_ = None // смещение
 self.cost_ = None //история штрафов в процессе обучения
 def net_input(self, X):
 return np.dot(X, self.w_) + self.w0_
 def activate(self, z):
 return z
 def theta(self, phi):
 return np.where( phi >= 0, 1, -1 )
 def predict(self, X):
 z = self.net_input(X)
 phi = self.activate(z)
 return self.theta(phi)
 def fit(self, X, y): // инициализация процесса обучения
 rgen = np.random.RandomState(self.random_state)
 self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
 self.w0_ = rgen.normal(loc=0.0, scale=0.01, size=None)
 self.cost_ = [] // обучаем
 for _ in range(0, self.n_iter):
 net_input = self.net_input(X)
 phi = self.activate(net_input)
 errors = ( y - phi )
 self.w_ += self.eta * X.T.dot(errors)
 self.w0_ += self.eta * errors.sum()
 cost = (errors**2).sum() / 2.0
 self.cost_.append(cost) // обучение закончено
 return self
adal_and=AdalineGD(0.01, 20, 1)
import numpy as np
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,0,0,1])
adal_and.fit(X,y)
<__main__.AdalineGD at 0x1f110949fa0>
X_val=np.array([[2,2], [-1,-2], [0,0.9], [0.9,0.9]])
adal_and.predict(X_val)
array([ 1, -1, 1, 1])
