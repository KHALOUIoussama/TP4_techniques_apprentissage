import numpy as np


class LinearClassifier(object):
    def __init__(self, x_train, y_train, x_val, y_val, num_classes, bias=False):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.bias = bias  # when bias is True then the feature vectors have an additional 1

        num_features = x_train.shape[1]
        if bias:
            num_features += 1

        self.num_features = num_features
        self.num_classes = num_classes
        self.W = self.generate_init_weights(0.01)

    def generate_init_weights(self, init_scale):
        return np.random.randn(self.num_features, self.num_classes) * init_scale

    def train(self, num_epochs=1, lr=1e-3, l2_reg=1e-4, lr_decay=1.0, init_scale=0.01):
        """
        Train the model with a cross-entropy loss
        Naive implementation (with loop)

        Inputs:
        - num_epochs: the number of training epochs
        - lr: learning rate
        - l2_reg: the l2 regularization strength
        - lr_decay: learning rate decay.  Typically a value between 0 and 1
        - init_scale : scale at which the parameters self.W will be randomly initialized

        Returns a tuple for:
        - training accuracy for each epoch
        - training loss for each epoch
        - validation accuracy for each epoch
        - validation loss for each epoch
        """
        loss_train_curve = []
        loss_val_curve = []
        accu_train_curve = []
        accu_val_curve = []

        self.W = self.generate_init_weights(init_scale)  # type: np.ndarray

        sample_idx = 0
        num_iter = num_epochs * len(self.x_train)
        for i in range(num_iter):
            # Take a sample
            x_sample = self.x_train[sample_idx]
            y_sample = self.y_train[sample_idx]
            if self.bias:
                x_sample = augment(x_sample)

            # Compute loss and gradient of loss
            loss_train, dW = self.cross_entropy_loss(x_sample, y_sample, l2_reg)

            # Take gradient step
            self.W -= lr * dW

            # Advance in data
            sample_idx += 1
            if sample_idx >= len(self.x_train):  # End of epoch

                accu_train, loss_train = self.global_accuracy_and_cross_entropy_loss(self.x_train, self.y_train, l2_reg)
                accu_val, loss_val, = self.global_accuracy_and_cross_entropy_loss(self.x_val, self.y_val, l2_reg)

                loss_train_curve.append(loss_train)
                loss_val_curve.append(loss_val)
                accu_train_curve.append(accu_train)
                accu_val_curve.append(accu_val)

                sample_idx = 0
                lr *= lr_decay

        return loss_train_curve, loss_val_curve, accu_train_curve, accu_val_curve

    def predict(self, X):
        """
        return the class label with the highest class score i.e.

            argmax_c W.X

         X: A numpy array of shape (D,) containing one or many samples.

         Returns a class label for each sample (a number between 0 and num_classes-1)
        """
        
        if (np.shape(X)[1] < np.shape(self.W)[0]) & self.bias:
            X = augment(X)
        
        class_label = np.zeros(X.shape[0])
        #############################################################################
        # TODO: Return the best class label.                                        #
        #############################################################################
        # Compute the class scores for the input data X
        scores = np.exp(np.dot(X, self.W))
        scores /= np.sum(scores, axis=1, keepdims=True)

        # Use np.argmax to get the index of the highest score
        # This index corresponds to the predicted class label
        class_label = np.argmax(scores, axis=1)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return class_label

    def global_accuracy_and_cross_entropy_loss(self, X, y, reg=0.0):
        """
        Compute average accuracy and cross_entropy for a series of N data points.
        Inputs:
        - X: A numpy array of shape (D, N) containing many samples.
        - y: A numpy array of shape (N) labels as an integer
        - reg: (float) regularization strength
        Returns a tuple of:
        - average accuracy as single float
        - average loss as single float
        """
        accu = 0.0
        loss = 0.0
        
        
        #############################################################################
        # TODO: Compute the softmax loss & accuracy for a series of samples X,y .   #
        #############################################################################
        if self.bias & (np.shape(X)[1] < np.shape(self.W)[0]):  # Check if the bias term should be used
                X = augment(X)
        
        num_samples = X.shape[0]

        for i in range(num_samples):
            x_sample = X[i]
            y_sample = y[i]
            loss_i, _ = self.cross_entropy_loss(x_sample, y_sample, reg)
            loss += loss_i
        
        # Calculate the average accuracy and loss
        accu = (self.predict(X) == y).mean()
        loss /= y.shape[0]

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return accu, loss


    def  cross_entropy_loss(self, x, y, reg=0.0):
        """
        Cross-entropy loss function for one sample pair (X,y) (with softmax)
        C.f. Eq.(4.104 to 4.109) of Bishop book.

        Input have dimension D, there are C classes.
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - x: A numpy array of shape (D,) containing one sample.
        - y: training label as an integer
        - reg: (float) regularization strength
        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
        """
          
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)
        
        #############################################################################
        # TODO: Compute the softmax loss and its gradient.                          #
        # Store the loss in loss and the gradient in dW.                            #
        # 1- Compute softmax => eq.(4.104) or eq.(5.25) Bishop                      #
        # 2- Compute cross-entropy loss => eq.(4.108)                               #
        # 3- Dont forget the regularization!                                        #
        # 4- Compute gradient => eq.(4.109)                                         #
        # To avoid numerical instability, subtract the maximum                      #
        # class score from all scores of a sample.                                  #
        #############################################################################

        
        # Compute the class scores for a sample x
        scores = x.dot(self.W)
        
        # Numerical stability fix: subtract the max score from all scores
        shift_scores = scores - np.max(scores)
        
        # Compute the softmax probabilities
        softmax_probs = np.exp(shift_scores) / np.sum(np.exp(shift_scores))
        
        # Compute the loss: -log of the probability of the correct class
        loss = -np.log(softmax_probs[y])
        
        # Regularization term: 0.5 * reg * sum(W^2)
        loss += 0.5 * reg * np.sum(self.W * self.W) 

         # Gradient calculation
        softmax_probs[y] -= 1  # subtract 1 from the probability of the correct class
        dW = x[:, np.newaxis] * softmax_probs[np.newaxis, :]  # outer product
        dW += reg * self.W  # add regularization gradient
        
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return loss, dW


def augment(x):
    if len(x.shape) == 1:
        return np.concatenate([x, [1.0]])
    else:
        return np.concatenate([x, np.ones((len(x), 1))], axis=1)
