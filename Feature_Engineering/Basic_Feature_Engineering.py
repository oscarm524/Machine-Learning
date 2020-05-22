####################################################
## Basic Feature Engineering Function (on inputs) ##
####################################################

## X: array of inputs 

def choose(n, r):
    
    return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

def pairwise_sum(X):
    
    ## Here we find the dimensions of X
    n_row = X.shape[0]
    n_col = X.shape[1]
    
    ## Here we find the number of pairwise sums
    sums = int(choose(n_col, 2))
    
    ## Here we define a list to store results
    X_out = []
    
    for i in range(0, sums-1):
        for j in range(i+1, sums):
            X_out.append(X[:, i] + X[:, j])
    
    ## Here we change the out to array
    X_out = np.array(X_out)
    X_out = np.transpose(X_out)
    
    return X_out
    

def pairwise_diff(X):
    
    ## Here we find the dimensions of X
    n_row = X.shape[0]
    n_col = X.shape[1]
    
    ## Here we find the number of pairwise sums
    sums = int(choose(n_col, 2))
    
    ## Here we define a list to store results
    X_out = []
    
    for i in range(0, sums-1):
        for j in range(i+1, sums):
            X_out.append(X[:, i] - X[:, j])
    
    ## Here we change the out to array
    X_out = np.array(X_out)
    X_out = np.transpose(X_out)
    
    return X_out

    
def pairwise_prod(X):
    
    ## Here we find the dimensions of X
    n_row = X.shape[0]
    n_col = X.shape[1]
    
    ## Here we find the number of pairwise sums
    sums = int(choose(n_col, 2))
    
    ## Here we define a list to store results
    X_out = []
    
    for i in range(0, sums-1):
        for j in range(i+1, sums):
            X_out.append(X[:, i] * X[:, j])
    
    ## Here we change the out to array
    X_out = np.array(X_out)
    X_out = np.transpose(X_out)
    
    return X_out
