# Clusterings
Clustering algorithms play a crucial role in data analysis and machine learning, enabling us to identify patterns, group similar data points, and gain insights from complex datasets. 
I devised from scratch K-Means, K-Means++, and Bisecting Hierarchical K-Means clustering algorithms, and compared their performance using metricsthe Silhouette Coefficient evaluation metrics.

## Overview

Clustering is the process of grouping similar data points together so that points in the same group are more alike than those in different groups. In this project, three clustering techniques were explored:

- **K-Means Clustering:**  
  A popular and simple algorithm that partitions data into *k* clusters by iteratively refining cluster centroids. The algorithm seeks to minimize the sum of squared Euclidean distances between data points and their assigned centroids.

- **K-Means++ Clustering:**  
  An enhancement over standard K-Means, K-Means++ improves centroid initialization. The algorithm chooses the first cluster centroid at random and then selects subsequent centroids with a probability proportional to the square of the distance from the nearest existing centroid. This tends to enhance convergence and clustering quality even though it slightly increases computational cost.

- **Bisecting K-Means Hierarchical Clustering:**  
  A hybrid approach combining K-Means with divisive hierarchical clustering. This method starts with a single cluster containing all data points and recursively splits a chosen cluster (typically the one with the largest sum of squared distances) into two sub-clusters using K-Means. This process is repeated until the desired number of clusters is reached.

A detailed pseudocode for each method is included in the source code.

---

## Algorithm Descriptions

### K-Means Clustering Algorithm

The K-Means algorithm works as follows:

1. **Initialization:**  
   Randomly choose *k* data points as the initial cluster centroids.

2. **Assignment Phase:**  
   Assign each data point to the nearest centroid based on squared Euclidean distance.

3. **Optimization Phase:**  
   Recompute the centroid of each cluster as the mean of all data points assigned to it.  
   Repeat the assignment and optimization steps until convergence (i.e. until cluster assignments no longer change significantly).

*Real-world analogy:*  
Imagine sorting clothing items by texture, colour, and size. Grouping them in piles helps you wash similar clothes together and prevent damage or staining—in the same way, K-Means groups similar data points so that they share representative centroids.

---

### K-Means++ Clustering Algorithm

K-Means++ refines the centroid initialization process:

1. **Initialization:**  
   - Uniquely select the first centroid at random.
   - For each remaining data point, compute its distance to the nearest already-chosen centroid.
   - Select subsequent centroids from the data with probability proportional to the square of these distances.
  
2. **Proceed:**  
   With all centroids initialized, continue with the standard K-Means iterations as described above.

This approach generally produces better initial centroids, leading to faster convergence and improved clustering quality.

---

### Bisecting K-Means Hierarchical Clustering Algorithm

This algorithm builds a hierarchical clustering tree by:

1. **Initialization:**  
   Start with the entire dataset as a single cluster (the root node).

2. **Recursive Splitting:**  
   Select the cluster (leaf node) with the largest within-cluster sum of squared distances.  
   Split the chosen cluster into two sub-clusters using the K-Means algorithm (with *k* set to 2).  
   Add the two sub-clusters as children in the clustering tree.

3. **Termination:**  
   Repeat the splitting process until a specified number of final clusters is obtained.

This method produces a dendrogram-like structure reflecting how data points are progressively divided into smaller, more similar groups.

---

## Performance Comparison

The evaluation of clustering performance was conducted using the silhouette coefficient, which measures how similar an object is to its own cluster compared to other clusters. The following observations were made:

### K-Means Clustering

- **Silhouette Coefficients by *k*:**
  - k = 2: 0.22217
  - k = 3: 0.18563
  - **k = 4: 0.22795** (best)
  - k = 5: 0.16091
  - k = 6: 0.15676
  - k = 7: 0.14494
  - k = 8: 0.13192
  - k = 9: 0.14579

The best silhouette coefficient is achieved when using 4 clusters (≈ 0.22795).

### K-Means++ Clustering

- **Silhouette Coefficients by *k*:**
  - k = 2: 0.22217
  - k = 3: 0.21682
  - k = 4: 0.19474
  - k = 5: 0.21061
  - k = 6: 0.13806
  - k = 7: 0.12152
  - k = 8: 0.12679
  - k = 9: 0.13449

For K-Means++, the highest score is seen at k = 2 (≈ 0.22217).

### Bisecting K-Means Hierarchical Clustering

- **Silhouette Coefficients by number of clusters (s):**
  - s = 2: 0.22217
  - s = 3: 0.21736
  - s = 4: 0.13559
  - s = 5: 0.15166
  - s = 6: 0.16843
  - s = 7: 0.16128
  - s = 8: 0.15119
  - s = 9: 0.14132

Here, the best silhouette score is also achieved using 2 clusters (≈ 0.22217).

### Overall Comparison

While K-Means++ and Bisecting K-Means yield their highest scores with only 2 clusters, the standard K-Means approach achieves its best performance with 4 clusters (silhouette coefficient ≈ 0.22795). Based on these evaluations, standard K-Means clustering with k = 4 delivers the best separation quality for the dataset under investigation.

---

## Observations and Conclusions

- **Algorithm Efficiency & Quality:**  
  Although K-Means++ provides improved initialization, its best performance was observed for fewer clusters compared to standard K-Means. Similarly, bisecting hierarchical clustering can generate a natural hierarchy but may not always optimize the silhouette coefficient as effectively.

- **Optimal Cluster Number:**  
  Standard K-Means achieves the highest silhouette coefficient when clustering the data into 4 groups, suggesting that these clusters are optimally separated given the dataset's intrinsic structure.

- **Real-World Applications:**  
  Clustering techniques like these have a broad range of applications, from customer segmentation in marketing and anomaly detection in financial analysis to optimizing delivery routes and organizing large-scale document collections.

- **Implementation Insight:**  
  While the pseudo code explains the iterative assignment and optimization processes for each algorithm, the actual code (see source files in the repository) provides a clear and practical demonstration of how these clustering methods work.

---

## Contributing

Contributions are welcome! If you have suggestions for further improvements or alternative approaches, please feel free to open an issue or submit a pull request.

---
[LICENSE](https://github.com/clareokafor/Clusterings?tab=MIT-1-ov-file)
