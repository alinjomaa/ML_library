import numpy as np

class MultipleLinearRegression:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def fit(self):
        X = np.array(self.x_data)
        y = np.array(self.y_data)

        ones = np.ones(len(X))
        X = np.column_stack((ones, X))  

        Xt = X.T

        XtX = np.dot(Xt, X)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(Xt, y)

        beta = np.dot(XtX_inv, Xty)
        return beta

    def predict(self, beta, x_new):
        x_new = np.array(x_new)
        x_new = np.insert(x_new, 0, 1)   
        return np.dot(x_new, beta)






        
    