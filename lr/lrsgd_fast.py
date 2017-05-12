import math 

class LogisticRegressionSGD_fast:
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = 0.09
        self.weight = [0.0] * n_feature
        self.temp = [0.0] * n_feature
        self.mu = 0.0
        self.size= n_feature
             
    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        newWeight = [0.0] * self.size
        w = [0.0] * len(X)
        val = self.predict_prob(X)        
        grad = [(y-1.0) * i[1] for i in X]        
        grad1 = float((math.exp(-math.fsum((self.weight[f]*v for f, v in X)))) * val)
        grad2 = [i[1] * -1 * grad1 for i in X]     
        for i in range(len(w)):
            w[i] = (grad[i] - grad2[i])
        
        w = [i*self.eta for i in w]
        for i in range(len(X)):
            newWeight[i] = self.weight[X[i][0]] -w[i]
        
        self.weight = newWeight[:]
    
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
