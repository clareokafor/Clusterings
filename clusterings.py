# OGOCHUKWU JANE OKAFOR

# CLUSTERINGS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Loading the data set using pandas
#dataset = pd.read_csv('/dataset', delimiter=' ', header=None)
dataset = pd.read_csv('/Users/mac/Downloads/dataset', delimiter=' ', header=None)


# Eliminating the columns (index 1) that have the string values (e.g elephant and co)
data = dataset.iloc[:, 1:] 

# Normalizing the data using Gaussian normalization
mean = data.mean() # data mean
std = data.std() # data standard deviation
data = (data - mean) / std # normalised data

# Converting the data to a numpy array
data = data.values

def euclideanDist(x, y):
    """
    This function will compute the Euclidean distance between vector x and vector y.
    Parmaters (Arguments):
            1. x  - 1-dimensional NumPy array representing the 1st vector.
            2. y  - 1-dimensional NumPy array representing the 2nd vector.
    Returns:
              - A float.
              - Represntsthe Euclidean distance between x and y.
    """
    # Computing the Euclidean distance between x and y
    return np.linalg.norm(x - y) 


def squaredEuclideanDist(x, y):
    """
    This function will compute the squared Euclidean distance between vector x and vector y.
    Parmaters (Arguments):
            1. x: 1-dimensional NumPy array representing the 1st vector.
            2. y: 1-dimensional NumPy array representing the 2nd vector.
    Returns:
              - A float.
              - Represntsthe Euclidean distance between x and y.
    """
    diff = x - y # Computing the difference between vectors x and y
    squared_diff = diff ** 2 # Squaring each vector of the difference array
    sq_euclidean_dist = np.sum(squared_diff)  # Sum the squared differences to get the squared Euclidean distance
    return sq_euclidean_dist


def KMeansClustering(data, k):
    """
    Utilizes the K-Means clustering algorithm to divide the input data into k clusters.

    Parameters:
        1. data:  A 2D array of shape (n_samples, n_features) containing the data to be clustered.
        2. k:   - int.
                - represents the number of clusters.
    Returns:
            numpy.ndarray.
                - cluster designations for each data point.
    """
    # Step 1: Initialization phase
    np.random.seed(42) # Setting a fixed seed
   
    # Choosing k random cluster representatives from the dataset
    init_cltrs_reps = np.random.choice(data.shape[0], size=k, replace=False)
    cltrs_reps = data[init_cltrs_reps] # The variable cltrs reps i other words centroids hold the selected data points.

    # Initializing cluster assignments to be the labels
    cltrs = np.zeros(data.shape[0], dtype=np.int32)

    # Iteratively refine cluster assignments and representatives
    prev_cltrs_reps = None # storing the cluster representatives from the previous iteration
    while not np.array_equal(cltrs_reps, prev_cltrs_reps): # runs until the cluster representatives stop changing between iterations.
        prev_cltrs_reps = np.copy(cltrs_reps)

    # Step 2: Assignment phase
    # Assign each object in data to its closest representative
        for n, x in enumerate(data):
            # Computing squared Euclidean distances between data point x and each cluster represenataive
            dists = [squaredEuclideanDist(x, clr) for clr in cltrs_reps]
            # Assigning data point x to the closest cluster represenataive
            cltrs[n] = np.argmin(dists)
            
    # Step 3: Optimization phase
    # Compute new cluster representatives as the means of the current clusters
        for m in range(k):
            # Get indices of data points assigned to cluster m
            reps = np.where(cltrs == m)[0]
            if reps.size > 0: # if the size of cluster representatives is greater than zero
                # Compute mean/average of data points assigned to cluster m and update representative
                cltrs_reps[m] = np.mean(data[reps], axis=0)
            else:
                # If no points assigned to cluster, reassign representativeto a random data point
                cltrs_reps[m] = data[np.random.choice(data.shape[0], size=1)]

    return cltrs # returns the final cluster assignments


def KMeansPPClustering(data, k):
    """
   Utilizes the K-Means++ clustering algorithm to divide the input data into k clusters.

    Parameters:
        1. data:  A 2D array of shape (n_samples, n_features) containing the data to be clustered.
        2. k:   - int.
                - represents the number of clusters.
    Returns:
            numpy.ndarray.
                - cluster designations for each data point.
    """
    # Setting a fixed seed to ensure reproducibility
    np.random.seed(42)

    # Step 1: Initialization phase
    # Uniquely selecting at random select the first representative Y1, from the Data
    representatives = np.zeros((k, data.shape[1]))
    selected_index = np.random.choice(data.shape[0], size=k, replace=False)
    representatives = data[selected_index]

    # Step 2: Assignment phase
    # Selecting the next centroid by computing the probability for each data point to be selected as representative.
    for i in range(1, k): # For each data point, compute the distance to the nearest centroid.
        distances = np.zeros((data.shape[0], i))
        for j in range(i):
            # Calculating the distance between each data point and centroid
            distances[:, j] = euclideanDist(data, representatives[j])
        
        # Computing the minimum distance for each data point and calculate the probability of each data point being selected as a representative.
        min_distances = np.min(distances, axis=1)
        probs =  min_distances ** 2 / np.sum(min_distances ** 2) # This probability is directly proportional to the distance from the nearest crepresentative squared.
        representatives[i] = data[np.random.choice(data.shape[0], p=probs)] # Selecting the next representative randomly based on the probability calculated

    # Step 3: Proceed with the standard k-means using Y1, …,Yk as initial cluster representatives
    # Initialize cluster assignments to be the labels
    kpp_cltrs = np.zeros(data.shape[0], dtype=np.int32)

    # Iteratively refine cluster assignments and representatives
    prev_cltrs_reps = None
    while not np.array_equal(kpp_cltrs, prev_cltrs_reps): # looping until the current cluster representatives are the same as the previous cluster representatives.
        prev_cltrs_reps = np.copy(kpp_cltrs) # copying the current cluster representatives to the previous cluster representatives.

        # Step 3(b)(i): Assign each object in data to its closest representative
        for n, x in enumerate(data): # iterating over the data points and their indices.
            dists = [squaredEuclideanDist(x, clr) for clr in representatives] # computing the squared Euclidean distances between the current data point and all cluster representatives.
            kpp_cltrs[n] = np.argmin(dists) # assigning the current data point to the closest cluster based on the computed distances.

        # Step 3(b)(ii): Update cluster representatives as the mean of the data points in the corresponding cluster
        for m in range(k): # iterating over the cluster indices.
            reps = np.where(kpp_cltrs == m)[0] # getting the indices of data points that belong to the current cluster.
            if reps.size > 0: # if the current cluster is full... 
                representatives[m] = np.mean(data[reps], axis=0) # update its representative as the mean of the data points in the cluster
    
    return kpp_cltrs # returns the final cluster assignments


def BisectingHierKMeans(data, s):
    """
    Utilizes the Bisecting Hierachical Kmeans clustering algorithm to divide the input data into s clusters.

    Parameters:
        1. data:  A 2D array of shape (n_samples, n_features) containing the data to be clustered.
        2. s:   - int.
                - represents the number of clusters.
    Returns:
          numpy.ndarray.
                - cluster designations for each data point.
    """
    # Initialising a tree to contain a single (root) vertex with entire dataset
    tree = [np.arange(data.shape[0])]

    # Bisecting the clusters until we have k clusters
    while len(tree) < s: # looping until the length of the cltrs list is less than the variable s.
        # Finding the cluster with the highest sum of squared distances
        ssds = [] # storing the sum of squared distances for each cluster in a empty list.
        for t in tree: # iterating over each cluster in the cltrs list.
            ssd = 0 # storing the sum of squared distances for a single cluster.
            for i in t: # iterating over each data point in a single cluster.
                ssd += np.sum((data[i] - data[t].mean(axis=0))**2) # the resulting value is added to the ssd variable.
            ssds.append(ssd) # appends the ssd variable, which represents the sum of squared distances for a single cluster to the ssds list.
        max_ssd_tree = tree.pop(np.argmax(ssds)) # the removed cluster is stored in the max_ssd_cltr variable.

        # Perform KMeans on the highest SSD cluster
        kmeans_cltr_labels = KMeansClustering(data[max_ssd_tree], 2)

        # Update the clusters with the KMeans results
        cltr1 = [max_ssd_tree[i] for i in range(len(max_ssd_tree)) if kmeans_cltr_labels[i] == 0] # cltr1 contains the elements of the list max_ssd_cltr
        cltr2 = [max_ssd_tree[i] for i in range(len(max_ssd_tree)) if kmeans_cltr_labels[i] == 1] # cltr2 contains the elements of the list max_ssd_cltr

        # Add as children of the current cluster
        tree.append(cltr1) # children of the current cluster 1.
        tree.append(cltr2) # children of the current cluster 2.

    # Assign cluster labels to each data point
    leaf_cltrs = np.zeros(data.shape[0], dtype=np.int32)
    for i, t in enumerate(tree): # loops through each cluster in cltrs 
      leaf_cltrs[t] = i # assigns the corresponding index of leaf_cltrs to the label of that cluster.
    # Return the leaf clusters
    return leaf_cltrs


def SilhouetteCoefficient(data, labels):
    """
    Identify the Silhouette Coefficient for the outcomes of clustering for the three algorithms.
    
    Parameters:
        1. data:    numpy.ndarray
                  - the dataset with shape number of samples, number of features.
        2. labels:  numpy.ndarray 
                  - the predicted labels for each sample.
    Returns:
        float: The average Silhouette Coefficient for the specified labels.
    """
    n = len(data)
    
    # Calculating distance matrix
    dist_mat = np.zeros((n, n)) # creating an empty distance matrix of size n x n
    for i in range(n): # iterating over each row in the distance matrix
        for j in range(n): # iterating over each column in the distance matrix
            dist_mat[i, j] = squaredEuclideanDist(data[i], data[j]) # calculating the squared Euclidean distance between data[i] and data[j] and store the result in the distance matrix
    
    # Computing Silhouette Coefficient for each data point
    sil_coeffs = np.zeros(n) # creating an empty array to store the Silhouette Coefficient for each data point
    for i in range(n): # iterating over each data point
        label_i = labels[i] # Get the label for data point i
        A_i = np.mean(dist_mat[i, labels == label_i]) # calculating the mean distance between data point i and all other data points with the same label
        B_i = np.inf # initialising the minimum mean distance between data point i and all other data points with a different label to infinity.

        for j in range(n): # iterating over each data point
            if labels[j] != label_i: # if the label for data point j is different from the label for data point i.
                B_ij = np.mean(dist_mat[i, labels == labels[j]]) # calculating the mean distance between data point i and all other data points with label j
                if B_ij < B_i: # if the mean distance between data point i and all other data points with label j is less than the current minimum
                    B_i = B_ij # updating the minimum mean distance between data point i and all other data points with a different label.

        sil_coeffs[i] = (B_i - A_i) / max(A_i, B_i) # calculating the Silhouette Coefficient for data point i and store the result in the sil_coeffs array
    
    # computing average Silhouette Coefficient
    score = np.mean(sil_coeffs) # Calculate the average Silhouette Coefficient across all data points
    return score # returns the average Silhouette Coefficient

plt.ion()
def SCValues(data):
    """
    This function computes Silhouette Scores for different clustering algorithms.
    Parameters:
            data: numpy.ndarray.
                - data to be clustered.
    Returns:
            tuple:
          - containing the Silhouette Scores for KMeans, KMeans++, and Bisecting Hierarchical KMeans.
    """
    # Defining a range of number of clusters to be tried, that is from [2,3,...,9]
    scorings = range(2, 10) 
    km = PlottingSCBarChart("KMeans", data, scorings) # computing Silhouette Scores for KMeans algorithm with different number of clusters.
    kpp = PlottingSCBarChart("KMeans++", data, scorings) # computing Silhouette Scores for KMeans++ algorithm with different number of clusters.
    bhm = PlottingSCBarChart("BisectingHierKMeans", data, scorings) # computing Silhouette Scores for Bisecting Hierarchical KMeans algorithm with different number of clusters.
    return km, kpp, bhm # returns the Silhouette Scores for all three clustering algorithms.

def PlottingSCBarChart(name, data, scorings):
    """
    Plots a bar chart of silhouette coefficients for different values of k, for a specified clustering algorithm.

    Parameters:
        1. name:  - string 
                    - the name of the clustering algorithm to use Must be one of "KMeans", "KMeans++", or "BisectingHierKMeans".
        2.data:   - numpy.ndarray.
                    - data to be clustered.
        3. scorings
                  - a list of integers representing the number of clusters to try.
    Returns:
            scores 
    """
    # Compute the silhouette coefficients for each value of k
    scores = []
    for score in scorings:
        if name == "KMeans":
            cltrs = KMeansClustering(data, score)  # performs KMeans clustering
        elif name == "KMeans++":
            cltrs = KMeansPPClustering(data, score)  # performs KMeans++ clustering
        elif name == "BisectingHierKMeans":
            cltrs = BisectingHierKMeans(data, score)  # performs Bisecting Hierarchical KMeans clustering
        coef = SilhouetteCoefficient(data, cltrs)  # computing the silhouette coefficient
        print(f"The silhouette coefficient ({name}) when cluster = {score} is: {coef}")
        scores.append(coef) # adds the silhouette coefficient in the scores empty list

    # Plot the silhouette coefficients as a bar chart
    fig, ax = plt.subplots()
    ax.bar(scorings, scores)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Coefficient')
    ax.set_title(f'{name} Silhouette Coefficient for Different Values')
    plt.show()

    return scores

km, kpp, bhm = SCValues(data)  # computing silhouette coefficients for KMeans, KMeans++, and Bisecting Hierarchical KMeans
