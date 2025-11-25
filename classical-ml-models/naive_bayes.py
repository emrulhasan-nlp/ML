"""Multinomial Naive Bayes Classifier Implementation"""

import numpy as np

class MultinomialNaivebayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_log_prior = None      # shape: (n_classes,)
        self.feature_log_prob = None     # shape: (n_classes, n_features)
    
    def fit(self, X, y):
        """
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # 1. Class prior P(y)
        class_counts = np.array([np.sum(y == c) for c in self.classes])
        self.class_log_prior = np.log(class_counts / n_samples)

        # 2. Feature counts per class: count(w_j, c)
        feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]                 # all samples of class c
            feature_counts[idx, :] = np.sum(X_c, axis=0)  # sum counts over docs

        # 3. Apply Laplace smoothing and compute log P(w_j | y=c)
        # Formula: P(w_j | c) = (count(w_j,c) + alpha) / (sum_k count(w_k,c) + alpha * V)
        V = n_features  # vocabulary size
        smoothed_counts = feature_counts + self.alpha   # count(w_j,c) + alpha
        smoothed_totals = feature_counts.sum(axis=1, keepdims=True) + self.alpha * V
        self.feature_log_prob = np.log(smoothed_counts / smoothed_totals)

        # return self
    
    def predict(self, X):
        """
        X: numpy array of shape (n_samples, n_features)
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        log_probs = np.zeros((n_samples, n_classes))

        # log P(y=c) + sum_j x_ij * log P(w_j | c)
        for idx, c in enumerate(self.classes):
            feature_log_prob_c = self.feature_log_prob[idx, :]  # (n_features,)
            log_probs[:, idx] = self.class_log_prior[idx] + X.dot(feature_log_prob_c)

        return self.classes[np.argmax(log_probs, axis=1)]


if __name__ == "__main__":
    # Example usage
    X_train = np.array([
        [2, 1, 0],
        [1, 0, 1],
        [2, 0, 1],
        [0, 1, 2],
        [1, 1, 1]
    ])
    y_train = np.array([0, 0, 0, 1, 1])

    model = MultinomialNaivebayes(alpha=1.0)
    model.fit(X_train, y_train)

    X_test = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [2, 1, 1]
    ])
    predictions = model.predict(X_test)
    print("Predictions:", predictions)
