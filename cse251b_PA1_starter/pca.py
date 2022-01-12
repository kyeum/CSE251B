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
    def __init__(self, x: np.ndarray, num_components):
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
        self.x = x


    def fit(self, X):
        """
        Set the internal parameters of the PCA object to the data.

        Parameters
        ----------
        X : np.array
            Training data to fit internal parameters.
        """

        pass


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

        pass


    def fit_transform(self, X):
        self.fit(X)
        pass 

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

    def PCA_Emmet(self) :
        #1. subtract the mean image from every image.
        self.mean_img = np.average(self.x, axis=0)
        msd = self.x - self.mean_img  # A = M x d ( M : number of Image, d : dimension of the Image(number of pixel)) = 2785 x  1024
        #2. construct co-variance matrix. #outer product of the two column vector d -> covar = A * A.T
        cov_matrix = np.matmul(msd, msd.T) # 2785 x 2785 = M x M ( M x d * d x M)
        #3. Find eigenvalue, eigen vector
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        #4. Sort the eigen vectors with the largest eigenvalue -> first principal component
        # -1) sorting eigen value and eigenvector
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[idx]
        # -2) the Avi’s are actually the eigenvectors of the original huge matrix C
        eigen_vectors = (np.matmul(msd.T, eigen_vectors)).T  # map origin from original one - A.v

        # -3) projection
        self.normalized_eig_vecs = eigen_vectors / np.linalg.norm(eigen_vectors, 2, axis=0)
        self.principal_eigen_vectors = self.normalized_eig_vecs[:self.num_components].T

        #5. divide by the standard deviation of the projections, which is the square root of the eigenvalue
        self.principal_sqrt_eigen_values = np.sqrt(eigen_values[:self.num_components])
        self.projected = np.matmul(msd, self.principal_eigen_vectors) / self.principal_sqrt_eigen_values
        
        return self.projected, self.mean_img,  self.principal_sqrt_eigen_values, self.principal_eigen_vectors
