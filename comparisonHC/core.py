import numpy as np
from sklearn.metrics import adjusted_rand_score

import time,itertools

def closest_clusters(clusters,kernel):
    """Returns the indices of the two clusters that are closest to each
    other in the list.
    Given a list of clusters, this method is deterministic.
    Parameters
    ----------
    clusters : list of (list of examples)
        A list containing at least two clusters.
    
    Returns
    -------
    i : int
        The index of the first of the two closest clusters.
    j : int
        The index of the second of the two closest clusters.
    
    """
    time_elapsed = 0
    time_start = time.process_time()
                    
    n_clusters = len(clusters)

    i,j = None,None
    
    score_best = -float("inf")

    for p in range(n_clusters):
        for q in range(p+1,n_clusters):
            kernel_pq = kernel[clusters[p],:][:,clusters[q]]

            score = np.mean(kernel_pq)
                        
            if score > score_best:
                i,j = p,q
                score_best = score

    time_end = time.process_time()
    time_elapsed += (time_end-time_start)
    
    return  i,j

class ComparisonHC:
    """ComparisonHC
    Parameters
    ----------
    similarity : numpy array
                 a symmetric matrix of size(n_examples,n_examples) having similarity values
    
    Attributes
    ----------
    similarity : numpy array
                 a symmetric matrix of size(n_examples,n_examples) having similarity values
    
    clusters : list of (list of examples), len (n_clusters)
        A list containing the initial clusters (list of
        examples). Initialized to the empy list until the fit method is
        called.
    n_clusters : int
        The number of initial clusters. Initialized to 0 until the fit
        method is called.
    n_examples : int
        The number of examples.
    dendrogram : numpy array, shape (n_clusters-1, 3)
        An array corresponding to the learned dendrogram. After
        iteration i, dendrogram[i,0] and dendrogram[i,1] are the
        indices of the merged clusters, and dendrogram[i,2] is the
        size of the new cluster. The dendrogram is initialized to None
        until the fit method is called.
        
    time_elapsed : float
        The time taken to learn the dendrogram. It includes the time
        taken by the linkage to select the next clusters to merge. It
        only records the time elapsed during the last call to fit.
    """
    def __init__(self,similarity,n_examples):
        self.similarity = similarity

        self.clusters = []
                
        self.n_clusters = 0

        self.n_examples = n_examples

        self.dendrogram = None
                
        self.time_elapsed = 0
            
    def fit(self,clusters):
        """Computes the dendrogram of a list of clusters.
        Parameters
        ----------        
        clusters : list of (list of examples), len (n_clusters)
            A list containing the initial clusters (list of examples).
        
        Returns
        -------
        self : object
        Raises
        ------
        ValueError
            If the initial partition has less that n_examples.
        """
        n_examples = sum([len(cluster) for cluster in clusters])
        if n_examples != self.n_examples:
            raise ValueError("The initial partition should have exactly n_examples.")
        
        time_start = time.process_time()
        
        self.clusters = clusters

        self.n_clusters = len(clusters)

        self.dendrogram = np.zeros((self.n_clusters-1,4))
        
        clusters_indices = list(range(self.n_clusters))

        clusters_copy = [[example for example in cluster] for cluster in self.clusters]

        for it in range(self.n_clusters-1):
            i,j = closest_clusters(clusters_copy,self.similarity)
            
            if i > j:
                i,j = j,i
            clusters_copy[i].extend(clusters_copy[j])
            del clusters_copy[j]

            self.dendrogram[it,0] = clusters_indices[i]
            self.dendrogram[it,1] = clusters_indices[j]
            self.dendrogram[it,2] = len(clusters_copy[i])

            clusters_indices[i] = self.n_clusters+it
            del clusters_indices[j]
                        
        time_end = time.process_time()
        self.time_elapsed = (time_end-time_start)
                
        return self

    def cost_dasgupta(self,similarities):
        """Computes the cost of the dendrogram as proposed by Dasgupta in 'A
        cost function for similarity-based hierarchical
        clustering'. The lower the cost the better the dendrogram is.
        This cost is based on the idea that similar examples should be
        merged earlier in the dendrogram.
        Parameters
        ----------
        similarities : numpy array, shape (n_examples, n_examples)
            A numpy array containing the similarities between all the
            examples.
        
        Returns
        -------
        cost : float
            The cost of the last dendrogram learned by the fit
            method. Lower is better.
        Raises
        ------
        RuntimeError
            If the dendrogram has not been lerned yet.
        """
        if self.dendrogram is None:
            raise RuntimeError("No dendrogram, the fit method should be called first.")
        
        cost = 0
        for cluster in self.clusters:
            for (example_i,example_j) in itertools.combinations(cluster,2):
                cost += len(cluster)*similarities[example_i,example_j]

        for ((cluster_index_i,cluster_i),(cluster_index_j,cluster_j)) in itertools.combinations(enumerate(self.clusters),2):
            cluster_size = self.dendrogram[self._get_iteration(cluster_index_i,cluster_index_j),2]
            
            for example_i in cluster_i:
                for example_j in cluster_j:
                    cost += cluster_size*similarities[example_i,example_j]

        return cost

    def _get_iteration(self,cluster_index_i,cluster_index_j):
        """Returns the iteration at which two clusters are merged.
        The indices cluster_index_i and cluster_index_j refer to
        cluster in the list of initial clusters.
        Parameters
        ----------
        cluster_index_i : int
            The index in clusters of the first cluster.
        cluster_index_j : int
            The index in clusters of the second cluster.
        Returns
        -------
        it : int
            The iteration at which the two clusters are merged. None
            if no such iteration is found.
        Raises
        ------
        RuntimeError
            If the dendrogram has not been lerned yet.
        """
        if self.dendrogram is None:
            raise RuntimeError("No dendrogram, the fit method should be called first.")
        
        for it in range(self.n_clusters-1):
            if self.dendrogram[it,0] == cluster_index_i or self.dendrogram[it,1] == cluster_index_i:
                cluster_index_i = self.n_clusters+it
            if self.dendrogram[it,0] == cluster_index_j or self.dendrogram[it,1] == cluster_index_j:
                cluster_index_j = self.n_clusters+it
            if cluster_index_i == cluster_index_j:
                return it
        else:
            return None

    def average_ARI(self,max_level,dendrogram_truth,clusters_truth=None):
        """Computes the score of the learned dendrogram in terms of Average
        Adjusted Rand Index as described in the main paper and
        compared to the ground truth dendrogram. The higher the score
        the better the dendrogram is.
        
        This score assumes that the learned hierarchy have levels
        which correspond to cuts in the dendrograms with given numbers
        of clusters. Here, we consider power of 2 levels, that is
        partiotions of the space in 2 clusters, 4 clusters, 8
        clusters, ... 2**max_level clusters.
        Parameters
        ----------
        max_level : int
            The number of levels to consider.
        dendrogram_truth : numpy array, shape (n_clusters-1, 3)
            The true dendrogram for the data.
        clusters_truth : list of (list of examples)
            The initial clusters used to generate dendrogram_truth. If
            None, the same initial clusters than fit are
            used. (Default: None).
        Returns
        -------
        score : float
            The score of the last dendrogram learned by the fit
            method. Higher is better.
        Raises
        ------
        RuntimeError
            If the dendrogram has not been lerned yet.
        """
        if self.dendrogram is None:
            raise RuntimeError("No dendrogram, the fit method should be called first.")

        if clusters_truth is None:
            clusters_truth = self.clusters
        
        score = []
        for level in range(1,max_level+1):
            k_clusters_truth = self._get_k_clusters(dendrogram_truth,clusters_truth,2**level)
            k_clusters = self._get_k_clusters(self.dendrogram,self.clusters,2**level)
            
            k_clusters_truth_labels = np.zeros((self.n_examples,))
            for cluster_index,cluster in enumerate(k_clusters_truth):
                k_clusters_truth_labels[np.array(cluster,dtype=int)] = cluster_index-1
                
            k_clusters_labels = np.zeros((self.n_examples,))
            for cluster_index,cluster in enumerate(k_clusters):
                k_clusters_labels[np.array(cluster,dtype=int)] = cluster_index-1
            
            score.append(adjusted_rand_score(k_clusters_truth_labels,k_clusters_labels))
            
        return score

    def _get_k_clusters(self,dendrogram,clusters,k):
        """Cuts a dendrogram of the initial clusters to obtain a partition of
        the space in exactly k clusters.
        If k is higher than the number of initial clusters, the
        initial clusters are returned.
        Parameters
        ----------
        dendrogram : numpy array, shape (n_clusters-1, 3)
            The dendrogram that should be used to obtain the
            partition.
        clusters : list of (list of examples)
            The initial clusters of the dendrogram.
        k : int
            The number of clusters in the partition.
        Returns
        -------
        k_clusters : list of (list of examples)
            The k_clusters that are merged last in the dendrogram.
        """
        n_clusters = len(clusters)
        if k >= n_clusters:
            return clusters

        if k < 2:
            return [[example_i for cluster in clusters for example_i in cluster]]

        k_clusters = [[example_i for example_i in cluster] for cluster in clusters]

        clusters_indices = list(range(n_clusters))
        
        for it in range(n_clusters-k):
            i = clusters_indices.index(dendrogram[it,0])
            j = clusters_indices.index(dendrogram[it,1])

            if i > j:
                i,j = j,i

            k_clusters[i].extend(k_clusters[j])
            del k_clusters[j]

            clusters_indices[i] = n_clusters+it
            del clusters_indices[j]
            
        return k_clusters