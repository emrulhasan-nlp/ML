########## K Nearest Neighbbour ##########
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5):
        self.k=k
        
    def fit(self, X_train, y_train):
        self.X_train=np.array(X_train)
        self.y_train=np.array(y_train)
        
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        
        return np.array(predictions)
        
    def _predict(self,x):
        # Euclidian distance
        distances=[np.linalg.norm(x-x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_n_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_n_labels).most_common(1)
        
        return most_common[0][0]
   
# Example usage
if __name__ == "__main__":
    # Sample data
    X_train = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]]
    y_train = [0, 0, 0, 1, 1, 1]

    # Create the model
    knn = KNN(k=5)
    knn.fit(X_train, y_train)

    # Make Prediction
    X_test = [[4, 5], [10, 10]]
    predictions = knn.predict(X_test)
    print("Predictions for KNN:", predictions)