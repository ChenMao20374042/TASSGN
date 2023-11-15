from sklearn.cluster import MiniBatchKMeans
import torch
import numpy as np
import random


class Labeler():

    def __init__(self, num_nodes, thresh=30, repeat=10):
        """
            Args:
                num_nodes: the number of nodes
                thresh: the thresh of binary cluster
                seed: random seed
        """
        self.num_nodes = num_nodes
        self.thresh = thresh
        self.label_counter = 0  
        self.repeat = repeat

    # Algorithm 1
    def binary_cluster(self, train_x, train_index, train_labels, val_x, val_index, val_labels):
        """ Algorithm 1. Binary Cluster.

         Args:
            train_x: the representation of training series, in shape (B, D), where B is the number of samples, D is the feature dimensions.
            train_index: in shape (B,), used as index.
            train_labels: in shape (B, 1), used to store generated labels.

        Returns:
            None. The generated labels would be stored into train_labels and val_labels directly.
        """

        if train_x.shape[0] <= self.thresh:
            train_labels[train_index] = self.label_counter
            val_labels[val_index] = self.label_counter
            self.label_counter += 1
        
        # undergo further binary cluster
        else:
            best_val_cluster = None
            min_val_distance = np.inf

            # repeat several times to get the best validation cluster
            for i in range(self.repeat):
                cluster = MiniBatchKMeans(n_clusters=2)
                cluster = cluster.fit(train_x)
                if val_x.shape[0] == 0: # no validation samples
                    best_val_cluster = cluster
                    break
                else:
                    val_distance = cluster.transform(val_x)
                    val_distance = np.min(val_distance, axis=-1)
                    val_distance = val_distance.mean()
                    if val_distance < min_val_distance:
                        min_val_distance = val_distance
                        best_val_cluster = cluster

            train_pred_labels = best_val_cluster.labels_
            train_mask_0, train_mask_1 = train_pred_labels == 0, train_pred_labels == 1

            if np.sum(train_mask_0) == 0 or np.sum(train_mask_1) == 0:
                train_labels[train_index] = self.label_counter
                val_labels[val_index] = self.label_counter
                self.label_counter += 1
                return

            if len(val_x) > 0:
                val_pred_labels = best_val_cluster.predict(val_x)
            else:
                val_pred_labels = np.empty([0,])
            val_mask_0, val_mask_1 = val_pred_labels == 0, val_pred_labels == 1

            self.binary_cluster(train_x[train_mask_0], train_index[train_mask_0], train_labels, 
                            val_x[val_mask_0], val_index[val_mask_0], val_labels)
            self.binary_cluster(train_x[train_mask_1], train_index[train_mask_1], train_labels, 
                            val_x[val_mask_1], val_index[val_mask_1], val_labels)



    
