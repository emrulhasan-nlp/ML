"""
K-nearest neighbour or KNN is an instance-based machine learning algorithm used for classification and regression tasks. It's a non-parametric algorithm.
This script implements the KNN algorithm from scratch.
"""

import numpy as np 
from collections import Counter
class KNN:
    def __init__(self, k=3):
        self.k=k #k is the number of nearest neibours
    
    #Fit the model with training data    
    def fit(self, X_train, y_train):
        self.X_train=X_train
        self.y_train=y_train
        
    
    #Calculate the distance between two vectors
    def _distance(self, x1, x2):
        """Euclidian Distance between two vectors"""
        return np.sqrt(np.sum((x2-x1)**2))
        
    # find neighbours
    def neighbours(self, x):
        """Calcuate the neighbours between x and the X_trains"""
        distances=[self._distance(x,x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        
        # find the labels from y_trian for k_indices
        neighbors=[self.y_train[i] for i in k_indices]
        
        return neighbors
        
    
    # Make prediction with test data
    def predict(self, x_test):
        
        y_pred=[Counter(self.neighbours(x)).most_common(1)[0][0] for x in x_test]
        
        return y_pred
    
# Test the model with sample dataset

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    np.random.seed(42)
    
    #prepare some sample dataset
    X=np.random.rand(200, 2)
    y=(X.sum(axis=1)>1).astype(int)  #  if x1+x2>1, y=1 else y=0.
    
    # Train test split
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.2)
    model=KNN()
    model.fit(X_train, y_train)
    
    y_pred=model.predict(X_test)
    
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy:.2f}")
