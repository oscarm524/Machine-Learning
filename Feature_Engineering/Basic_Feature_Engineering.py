####################################################
## Basic Feature Engineering Function (on inputs) ##
####################################################

## X: array of inputs 

def choose(n, r):
    
    return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

def pairwise_sum(X):
    
    ## Here we find the number of columns of X
    n_col = X.shape[1]
    
    ## Here we find the number of pairwise sums
    sums = int(choose(n_col, 2))
    
    ## Here we define a list to store results
    X_out = []
    
    for i in range(0, sums-1):
        for j in range(i+1, sums):
            X_out.append(X[:, i] + X[:, j])
    
    ## Here we change the list of results to array
    X_out = np.array(X_out)
    X_out = np.transpose(X_out)
    
    return X_out
    

def pairwise_diff(X):
    
    ## Here we find the number of columns of X
    n_col = X.shape[1]
    
    ## Here we find the number of pairwise differences
    sums = int(choose(n_col, 2))
    
    ## Here we define a list to store results
    X_out = []
    
    for i in range(0, sums-1):
        for j in range(i+1, sums):
            X_out.append(X[:, i] - X[:, j])
    
    ## Here we change the list of results to array
    X_out = np.array(X_out)
    X_out = np.transpose(X_out)
    
    return X_out

    
def pairwise_prod(X):
    
    ## Here we find the number of columns of X
    n_col = X.shape[1]
    
    ## Here we find the number of pairwise prods
    sums = int(choose(n_col, 2))
    
    ## Here we define a list to store results
    X_out = []
    
    for i in range(0, sums-1):
        for j in range(i+1, sums):
            X_out.append(X[:, i] * X[:, j])
    
    ## Here we change the list of results to array
    X_out = np.array(X_out)
    X_out = np.transpose(X_out)
    
    return X_out
    
    
def squared_features(X):
    
    ## Here we find the number columns
    n_col = X.shape[1]
    
    ## Here we define a list to store results
    X_out = []
    
    for i in range(n_col):
        X_out.append(X[:, i] ** 2)
    
    ## Here we change the out to array
    X_out = np.array(X_out)
    X_out = np.transpose(X_out)
    
    return X_out


def cubed_features(X):
    
    ## Here we find the number columns
    n_col = X.shape[1]
    
    ## Here we define a list to store results
    X_out = []
    
    for i in range(n_col):
        X_out.append(X[:, i] ** 3)
    
    ## Here we change the out to array
    X_out = np.array(X_out)
    X_out = np.transpose(X_out)
    
    return X_out
