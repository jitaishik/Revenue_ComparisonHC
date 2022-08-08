import numpy as np
import sklearn.metrics

def get_AddS_triplets(oracle,n_examples):
    """Returns a symmetric similarity matrix representing the similarities
    between all the examples using the AddS triplets approach.
    Parameters
    ----------
    oracle : OracleTriplets
        An oracle used to query the comparisons. It should exhibit a
        method get_AddS_comparisons() that returns a scipy csr matrix,
        shape(n_examples**2,n_examples) A scipy csr_matrix containing
        values in {1,-1,0}. Given i!=j,k, in entry (i*n_examples+j,k),
        the value 1 indicates that the triplet (i,j,k) is available,
        the value -1 indicates that the triplet (i,k,j) is available,
        and the value 0 indicates that neither of the triplets is
        available.
    n_examples : int
        The number of examples handled by the oracle.
    Returns
    -------
    kernel : numpy array, shape (n_examples,n_examples)
        A nummpy array of similarities between the examples.
    """
    kernel = np.zeros((n_examples,n_examples))
    
    comps = oracle.get_AddS_comparisons()
    entries = comps.sum(axis=1).A1

    indices = np.arange(n_examples**2)
    i = indices//n_examples
    j = indices - i*n_examples
    
    kernel[i,j] = entries
    
    kernel += kernel.transpose()
    
    return kernel

def get_MulK_triplets(oracle,n_examples):
    """Returns a symmetric similarity matrix representing the similarities
    between all the examples using the MulK triplets approach.
    Parameters
    ----------
    oracle : OracleTriplets
        An oracle used to query the comparisons. It should exhibit a
        method get_MulK_comparisons() that returns a scipy csr matrix
        of shape(n_examples,(n_examples choose 2)) containing values
        in {1,-1,0}. Given i!=j,k, j<k, in entry (i,self._get_k(j,k)),
        the value 1 indicates that the triplet (i,j,k) is available,
        the value -1 indicates that the triplet (i,k,j) is available,
        and the value 0 indicates that neither of the triplets is
        available.
    n_examples : int
        The number of examples handled by the oracle.
    Returns
    -------
    kernel : numpy array, shape (n_examples,n_examples)
        A nummpy array of similarities between the examples.
    """
    kernel = np.zeros((n_examples,n_examples))
    
    comps = oracle.get_MulK_comparisons()
    kernel = comps.dot(comps.transpose())
    
    norms = np.sqrt(comps.getnnz(axis=1))
    norms = norms.reshape(-1,1) @ norms.reshape(1,-1)
    norms = np.where(norms == 0,1,norms) # This is to avoid issues with the true divide when the norm is 0 for i or j
    
    kernel = kernel.toarray()/norms
    np.fill_diagonal(kernel,0)

    return kernel

def tSTE_grad(X, n, no_dims, triplets, lbda, alpha, use_log):

    # Compute Student-t kernel
    sum_X = np.sum(np.square(X),axis=1,keepdims=True)
    base_K = 1 + (sum_X + (sum_X.transpose() - 2*X.dot(X.transpose()))) / alpha
    K = np.power(base_K,-(alpha+1)/2)
        
    # Compute value of cost function
    P = K[triplets[:,[0]],triplets[:,[1]]] / (K[triplets[:,[0]],triplets[:,[1]]] + K[triplets[:,[0]],triplets[:,[2]]])
    if use_log:
        C = -np.sum(np.log(np.where(P>0,P,0))) + lbda*np.sum(np.square(X))
    else:
        C = -np.sum(P) + lbda*np.sum(np.square(X))

    # Compute gradient
    G = np.zeros((n,no_dims))
    base_K = 1 / base_K
    for i in range(no_dims):
        const = (alpha + 1) / alpha
        if use_log:
            top = -const*((1-P)*base_K[triplets[:,[0]],triplets[:,[1]]]*(X[triplets[:,[0]],i]-X[triplets[:,[1]],i]) - (1-P)*base_K[triplets[:,[0]],triplets[:,[2]]]*(X[triplets[:,[0]],i]-X[triplets[:,[2]],i]))
            mid = -const*(-(1-P))*base_K[triplets[:,[0]],triplets[:,[1]]]*(X[triplets[:,[0]],i]-X[triplets[:,[1]],i])
            bot = -const*(1-P)*base_K[triplets[:,[0]],triplets[:,[2]]]*(X[triplets[:,[0]],i]-X[triplets[:,[2]],i])
            G[:,i] = np.bincount(np.vstack((triplets[:,[0]],triplets[:,[1]],triplets[:,[2]])).ravel(),weights=np.vstack((top,mid,bot)).ravel())
        else:
            top = -const*(P*(1-P)*base_K[triplets[:,[0]],triplets[:,[1]]]*(X[triplets[:,[0]],i]-X[triplets[:,[1]],i]) - P*(1-P)*base_K[triplets[:,[0]],triplets[:,[2]]]*(X[triplets[:,[0]],i]-X[triplets[:,[2]],i]))
            mid = -const*(-P*(1-P))*base_K[triplets[:,[0]],triplets[:,[1]]]*(X[triplets[:,[0]],i]-X[triplets[:,[1]],i])
            bot = -const*P*(1-P)*base_K[triplets[:,[0]],triplets[:,[2]]]*(X[triplets[:,[0]],i]-X[triplets[:,[2]],i])
            G[:,i] = np.bincount(np.vstack((triplets[:,[0]],triplets[:,[1]],triplets[:,[2]])).ravel(),weights=np.vstack((top,mid,bot)))
    G = -G + 2*lbda*X
    return C,G

def tSTE(triplets, n, no_triplets, no_dims=2, lbda=0, alpha=None, use_log=True):
    if alpha is None:
        alpha = no_dims-1

    # Initialize some variables
    X = np.random.randn(n, no_dims)*1e-4
    C = float("Inf")
    tol = 1e-7 # Convergence tolerance
    max_iter = 1000 # Maximum number of iterations
    eta = 2 # Learning rate
    best_C = C # Best error obtained so far
    best_X = X # Best embedding found so far

    # Perform main learning iterations
    it = 0
    no_incr = 0
    while it < max_iter and no_incr < 5:

        # Computer value of slack variables, cost function and gradient
        old_C = C
        C, G = tSTE_grad(X, n, no_dims, triplets, lbda, alpha, use_log)
               
        # Maintain best solution found so far
        if C < best_C:
            best_C = C
            best_X = X

        # Perform gradient update
        X = X - (eta / no_triplets * n) * G

        # Update learning rate
        if old_C > C + tol:
            no_incr = 0
            eta = eta * 1.01
        else:
            no_incr = no_incr + 1
            eta = eta * 0.5

        # End of iteration
        it += 1

    # Return best embedding
    return best_X

def get_tSTE_triplets(oracle,n_examples):
  X = tSTE(oracle.get_tSTE_comparisons(),n_examples,oracle.get_tSTE_comparisons().shape[0])

  kernel = sklearn.metrics.pairwise.cosine_similarity(X, dense_output=True)

  return kernel

def get_AddS_quadruplets(oracle,n_examples):
    """Returns a symmetric similarity matrix representing the similarities
    between all the examples using the AddS quadruplets approach.
    Parameters
    ----------
    oracle : OracleQuadruplets
        An oracle used to query the comparisons. It should exhibit a
        method comparisons_to_ref(i,j) which return all the
        comparisons associated with the pair (i,j) in a sparse matrix
        where in entry (k,l), the value 1 indicates that the
        quadruplet (i,j,k,l) is available, the value -1 indicates that
        the quadruplet (k,l,i,j) is available, and the value 0
        indicates that neither of the quadruplets is available.
    n_examples : int
        The number of examples handled by the oracle.
    Returns
    -------
    kernel : numpy array, shape (n_examples,n_examples)
        A nummpy array of similarities between the examples.
    """
    kernel = np.zeros((n_examples,n_examples))
    
    comps = oracle.get_AddS_comparisons()
    entries = comps.sum(axis=1).A1
    i,j = oracle._get_ij(np.arange((n_examples*(n_examples-1))//2))
    kernel[i,j] = entries
    
    kernel += kernel.transpose()
    
    return kernel

def get_MulK_quadruplets(oracle,n_examples):
    """Returns a symmetric similarity matrix representing the similarities
    between all the examples using the MulK quadruplets approach.
    Parameters
    ----------
    oracle : OracleQuadruplets
        An oracle used to query the comparisons. It should exhibit a
        method comparisons_to_ref(i,j) which return all the
        comparisons associated with the pair (i,j) in a sparse matrix
        where in entry (k,l), the value 1 indicates that the
        quadruplet (i,j,k,l) is available, the value -1 indicates that
        the quadruplet (k,l,i,j) is available, and the value 0
        indicates that neither of the quadruplets is available.
    n_examples : int
        The number of examples handled by the oracle.
    Returns
    -------
    kernel : numpy array, shape (n_examples,n_examples)
        A nummpy array of similarities between the examples.
    """
    kernel = np.zeros((n_examples,n_examples))
    
    comps = oracle.get_MulK_comparisons()
    kernel = comps.dot(comps.transpose())
    kernel = kernel.toarray()
    np.fill_diagonal(kernel,0)

    return kernel