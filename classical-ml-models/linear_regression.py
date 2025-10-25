"""
Linear Regeression is the simpliest statistical method that tries to find a linear reletionship between input features and the continuous output y and it is defined as 
y_pred=w.T*X+b where w and b are the weights and bias. This script implements the linear regression from scratch
"""
import numpy as np
class LinearRegression:
    def __init__(self, lr=0.01, num_iter=100):
        self.w=None
        self.b=None
        self.lr=lr
        self.num_iter=num_iter
        self.losess = []
        
    def cost(self,y_hat, y):
        """Calculate the cost"""
        m=len(y)
        
        cost=1/(2*m)*np.sum((y_hat-y)**2)
        
        return cost
    #Define training function
    
    def fit(self, X, y):
        
        m,n=X.shape  
        self.w=np.zeros(n)
        self.b=0
        for _ in range(self.num_iter):
            
            y_hat=np.dot(X, self.w)+self.b
            
            dw=(1/m)* np.dot(X.T, (y_hat-y))
            db= (1/m) * np.sum(y_hat-y)
            
            self.w -=self.lr*dw
            self.b-=self.lr*db
            loss=self.cost(y_hat, y)
            self.losess.append(loss)
    
    def predict(self,X_test):
        y_pred=np.dot(X_test, self.w)+self.b
        
        return y_pred
    
    
# Model training and testing
if __name__ == "__main__":
    # Generate 200 example with with random 2 features
    from sklearn.model_selection import train_test_split
    np.random.seed(42)
    X = np.random.rand(200, 2)
    y = X.sum(axis=1)
    
    # Train test split
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.2)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate test accuracy
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred)**2)
    print(f"Test MSE: {mse:.4f}")
    
    # Plot the train loss vs number of iteration
    import matplotlib.pyplot as plt
    plt.plot(model.losess)
    plt.title("Changes of cost during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

    