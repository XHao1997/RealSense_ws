import numpy as np
def preprocess2PCA(result_masks:list)->list:
    """
    The function preprocess2PCA takes a list of masks, finds the coordinates where
    the mask value is 1, transposes and flips the coordinates, and returns a list of
    these modified coordinates.
    
    :param result_masks: A list of binary masks representing the results of some
    image processing task. Each mask is a 2D numpy array where the value 1 indicates
    the presence of an object and 0 indicates the background
    :type result_masks: list
    :return: The function preprocess2PCA is returning a list of numpy arrays, where
    each array contains the coordinates (x, y) of the points where the mask value is
    equal to 1 in each mask from the input list result_masks.
    """
    xy_list = []
    for mask in result_masks:
        xy = np.where(mask==1)
        xy = np.array(xy).transpose()
        xy = np.fliplr(xy)
        xy_list.append(xy)
    return xy_list



def simplePCA(arr):
    """
    The function `simplePCA` calculates the mean, centers the data, 
    computes the covariance matrix, and returns the mean, eigenvalues, 
    and eigenvectors for Principal Component Analysis (PCA).
    
    :param arr: 3*n array [x,y,z]
    :return: The function `simplePCA` returns the mean of the input 
    array `arr`, the eigenvalues, and the eigenvectors calculated using 
    Principal Component Analysis(PCA).
    """

    # calculate mean
    m = np.mean(arr, axis=0)

    # center data
    arrm = arr-m

    # calculate the covariance, decompose eigenvectors and eigenvalues
    # M * vect = eigenval * vect
    # cov = M*M.T
    Cov = np.cov(arrm.T)
    eigval, eigvect = np.linalg.eig(Cov.T)

    # return mean, eigenvalues, eigenvectors
    return m, eigval, eigvect