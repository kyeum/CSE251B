import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """
    This class handles all things related to PCA for PA1.

    You can add any new parameters you need to any functions. This is an 
    outline to help you get started.

    You should run PCA on both the training and validation / testing datasets 
    using the same object.

    For the visualization of the principal components, use the internal 
    parameters that are set in `fit`.
    """
    def __init__(self, num_components):
        """
        Setup the PCA object. 

        Parameters
        ----------
        num_components : int
            The number of principal components to reduce to.
        Return
        ----------    
        """
        self.num_components = num_components

    
    def fit(self, X):
        """
        Set the internal parameters of the PCA object to the data.

        Parameters
        ----------
        X : np.array
            Training data to fit internal parameters.
        """
        #1. subtract the mean image from every image.
        self.mean_img = np.average(X, axis=0)
        msd = X - self.mean_img  # A = M x d ( M : number of Image, d : dimension of the Image(number of pixel)) = 2785 x  1024
        #2. construct co-variance matrix. #outer product of the two column vector d -> covar = A * A.T
        # Use Turk Pentland Trick (A^T * A)
        cov_matrix = np.dot(msd.T, msd) # 1024 x 1024 = d x d ( d x M * M x d)
        # cov_matrix = np.cov(msd.T)
        #3. Find eigenvalue, eigen vector
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        # Get eigenvectors of A
        # u_i = A * v_i
        eigen_vectors = np.dot(msd, eigen_vectors)

        #4. Sort the eigen vectors with the largest eigenvalue -> first principal component
        # -1) sorting eigen value and eigenvector
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:,idx]

        # -2) the Avi’s are actually the eigenvectors of the original huge matrix C
        # # map vector from original
        eigen_vectors = (np.matmul(msd.T, eigen_vectors)).T  # M x d
        # -3) projection
        #self.normalized_eig_vecs = eigen_vectors / np.linalg.norm(eigen_vectors, 2, axis=0)
        norm = np.sqrt(np.sum((eigen_vectors)**2, axis=-1)).reshape(-1,1)  # M
        self.normalized_eig_vecs = eigen_vectors / norm # M x d

        self.principal_eigen_vectors = self.normalized_eig_vecs[:self.num_components].T

        #5. divide by the standard deviation of the projections, which is the square root of the eigenvalue
        self.principal_sqrt_eigen_values = np.sqrt(eigen_values[:self.num_components])

    def transform(self, X):
        """
        Use the internal parameters set with `fit` to transform data.

        Make sure you are using internal parameters computed during `fit` 
        and not recomputing parameters every time!

        Parameters
        ----------
        X : np.array - size n*k
            Data to perform dimensionality reduction on

        Returns
        -------
            Transformed dataset with lower dimensionality
        """
        X = X - self.mean_img
        # Project
        projected = np.matmul(X, self.principal_eigen_vectors) / self.principal_sqrt_eigen_values
        
        return projected


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def plot_PC(self):
        '''
        Plot top 4 principal components
        the eigenvector with the largest eigenvalue is the first principal component,
        '''
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(8, 8)
        fig.set_dpi(100)
        axs[0, 0].set_title('PC 1')
        axs[0, 0].imshow(self.principal_eigen_vectors.T[0].real.reshape((32, 32)))
        axs[0, 1].set_title('PC 2')
        axs[0, 1].imshow(self.principal_eigen_vectors.T[1].real.reshape((32, 32)))
        axs[1, 0].set_title('PC: 3')
        axs[1, 0].imshow(self.principal_eigen_vectors.T[2].real.reshape((32, 32)))
        axs[1, 1].set_title('PC: 4')
        axs[1, 1].imshow(self.principal_eigen_vectors.T[3].real.reshape((32, 32)))
        plt.show()       

    def PCA_Emmet(self, X) :
        #1. subtract the mean image from every image.
        self.mean_img = np.average(X, axis=0)
        msd = X - self.mean_img  # A = M x d ( M : number of Image, d : dimension of the Image(number of pixel)) = 2785 x  1024
        #2. construct co-variance matrix. #outer product of the two column vector d -> covar = A * A.T
        cov_matrix = np.dot(msd, msd.T) # 2785 x 2785 = M x M ( M x d * d x M)
        #3. Find eigenvalue, eigen vector
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        #4. Sort the eigen vectors with the largest eigenvalue -> first principal component
        # -1) sorting eigen value and eigenvector
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:,idx]

        # -2) the Avi’s are actually the eigenvectors of the original huge matrix C
        # # map vector from original
        eigen_vectors = (np.matmul(msd.T, eigen_vectors)).T  # M x d
        print(np.shape(eigen_vectors))
        # -3) projection
        #self.normalized_eig_vecs = eigen_vectors / np.linalg.norm(eigen_vectors, 2, axis=0)
        norm = np.sqrt(np.sum((eigen_vectors)**2, axis=-1)).reshape(-1,1)  # M
        self.normalized_eig_vecs = eigen_vectors / norm # M x d

        self.principal_eigen_vectors = self.normalized_eig_vecs[:self.num_components].T

        #5. divide by the standard deviation of the projections, which is the square root of the eigenvalue
        self.principal_sqrt_eigen_values = np.sqrt(eigen_values[:self.num_components])
        
        #projection
        self.projected = np.matmul(msd, self.principal_eigen_vectors) / self.principal_sqrt_eigen_values
        
        return self.projected

    def PCA_generate(self, data) :
        
        msd = data - self.mean_img  # A = M x d ( M : number of Image, d : dimension of the Image(number of pixel)) = 2785 x  1024
        return np.matmul(msd, self.principal_eigen_vectors) / self.principal_sqrt_eigen_values




