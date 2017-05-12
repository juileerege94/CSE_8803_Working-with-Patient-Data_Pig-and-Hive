# Do not use anything outside of the standard distribution of python
# when implementing this class
import math 

class LogisticRegressionSGD:
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = 0.001
        self.weight = [0.0] * n_feature
        self.temp = [0.0] * n_feature
        self.mu = 0.0
        self.fsize= n_feature
             
    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        x = [0.0]*self.fsize
        for i in range(0,len(X)):
            x[X[i][0]] = X[i][1]
    
        grad = [(y-1.0) * i for i in x]
        val = self.predict_prob(X)
        grad1 = float((math.exp(-math.fsum((self.weight[f]*v for f, v in X)))) * val)
        grad2 = [i * -1 * grad1 for i in x]     
        mu_term = [i * 2 * self.mu for i in self.weight]
        
        for i in range(len(self.weight)):
            self.temp[i] = (grad[i] - grad2[i]) - mu_term[i]
        update = [i * self.eta for i in self.temp]
        self.weight = [(self.weight[i] + update[i]) for i in range(len(self.weight))]

        pass

    def predict(self, X):
        """
        Predict 0 or 1 given X and the current weights in the model
        """
        return 1 if self.predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        """
        Sigmoid function
        """
        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))
