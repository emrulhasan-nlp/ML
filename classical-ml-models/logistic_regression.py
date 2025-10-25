"""
Logistic Regression implementation from scratch. The logistic function is defined as sigmoid (z)=1/(1+e^-z).
"""

import numpy as np 

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100):
        self.w=None
        self.b=0
        self.num_iter=num_iter
        self.lr=lr
    
    #Define logistic function    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    # Define Cost function
    def cost(self, h, y):
        """Calcualte Cross entropy loss"""
        m=len(y)
        loss=- (1/m)*np.sum(y*np.log(h)+ (1-y)*np.log(1-h)) # - (1/m) sum (y log h)+ (1-y) *log(1-h) # h is the prediction
        
        return loss
        
    def fit(self, X, y):
        """ Training logistic regression model"""
        m,n =X.shape
        
        self.w=np.zeros(n)
        print(self.w)
        
        for _ in range(self.num_iter):
            
            z=np.dot(X, self.w) +self.b # z=w.T*X+b
            
            h=self.sigmoid(z)
            
            dw=(1/m)*np.dot(X.T, (h-y))
            
            db=(1/m)*np.sum(h-y)
            
            self.w -=self.lr *dw  
            self.b -=self.lr*db
 
    def predict(self, X):
        """ Prediction function"""
        # print(self.w) 
        # print(self.b)  
        z=np.dot(X,self.w)+self.b 
        output=self.sigmoid(z)
        
        return np.where(output >= 0.5, 1, 0) # if output>=.5, return 1 eslse 0
            

###########Train and test###########

if __name__ == "__main__":
    # Generate 200 example with with random 2 features
    from sklearn.model_selection import train_test_split
    np.random.seed(42)
    X = np.random.rand(200, 2)
    y = (X.sum(axis=1) > 1).astype(int) #  if x1+x2>1, y=1 else y=0.
    
    # Train test split
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.2)
    
    # Fit the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Calculate test accuracy
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Test accuracy: {accuracy:.2f}")


    