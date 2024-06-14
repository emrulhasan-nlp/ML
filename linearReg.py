import numpy as np 

"""
Formular for simple linear regression is y=wx+c where x, and y are the independent and dependent variables respectively. w represents the slope and c is the intercepts

w=dx/dy
"""

class LinearReg:
    
    def __init__(self):
        
        self.W=None
        self.c=None
        
    def fit(self, X, y):
        N=len(X)
        
        X_mean=np.mean(X)
        y_mean=np.mean(y)
        
        numer=0
        deno=0
        
        for i in range(N):
            numer +=(X[i]-X_mean)*(y[i]-y_mean)
            deno +=(X[i]-X_mean)**2
        
        self.W=numer/deno
        
        self.c=y_mean-(self.W*X_mean)
        
    def predict(self, X):
        y_preds=[]
        
        for x in X:
            y_pred=self.W*x +self.c
            y_preds.append(np.round(y_pred,4))
            
        return y_preds


# if __name__=='__main__':
#     X = np.array([3, 4, 6, 8, 10])
#     y = np.array([2, 4, 5, 4, 5])
#     model = LinearReg()
#     model.fit(X, y)

#     y_preds = model.predict(X)
#     print(y_preds)  # Output: [2.8, 3.4, 4.0, 4.6, 5.2]
    
############# Linear Regression with gradient descent#############

class LinearRegression:
    def __init__(self, epochs=100, lr=0.01):
        self.epochs=epochs
        self.lr=lr
        
        self.W=None
        self.c=None
    
    def fit(self, X, y):
        num_examples, num_feature=X.shape
        
        self.W =np.zeros(num_feature)
        
        self.c=0
        
        # Gradient Descent: a:=a- alpha d/dw (j(a))
        # a:=a-alpha sum (y_pred-y).X
        for epoch in range(self.epochs):
            y_pred=np.dot(X, self.W)+self.c
            
            dw=(1/num_examples)* np.dot(X.T, (y_pred-y))
            db=(1 / num_examples) * np.sum( y_pred - y)
                       
            # Update weights and bias
            self.W -= self.lr * dw
            self.c -= self.lr * db
    
    def predict(self, X):
        y_preds=np.round(np.dot(X, self.W)+self.c, 4)
        return y_preds
    

# Test run
if __name__ == "__main__":
    # Example data
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([6, 8, 10, 12])

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)
    print(f"Predictions: {predictions}")