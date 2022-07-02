# Intro to K-Means

K-means is an unsurpervised machine learning model, the objective of `K-means` is simply group similar data points together and discover the underlying patterns. To achieve this `K-means` looks for a fixed number (K) of clusters in a dataset.

In otherwords `K-means` algo identifies the _K_ number of centroids, and allocates every data point to the nearest cluster, hile keeping the centroids as small as possible.

The `means` in the `K-means` refers to averaging the data, that's finding the centroid.


[copied @ https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1]


## Determining the no. of clusters

There are several methods to determine the optimum number of clusters a few are:

  *  Elbow method
  *  Average Silhouette