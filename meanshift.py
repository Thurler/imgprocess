from PIL import Image
from pyimage import PyImage

import numpy as np
import scipy as sp
from scipy import spatial


class MeanShift(object):

    '''This class...'''

    def __init__(self):

        self.original = None
        self.msFilter = None
        self.msSegment = None

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFile(self, filepath):

        '''This function should open the image contained in the specified file
        and reset the class' variables'''

        # Reset variables
        self.msFilter = None
        self.msSegment = None

        # Load image
        self.original = PyImage()
        self.original.loadFile(filepath)

    def loadImage(self, image):

        '''This function should open the image contained in the specified file
        and reset the class' variables'''

        # Reset variables
        self.msFilter = None
        self.msSegment = None

        # Load image
        self.original = PyImage()
        self.original.loadImage(image)

    def saveFile(self, filepath):

        '''This function should save to disk the class' current state, by
        saving not only the original file, but also the meanshifted ones, if
        they exist.'''

        # Separate name from extension
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        # Save original
        self.original.saveFile(path + "original" + extension)

        # Save meanshift filter
        if self.msFilter is not None:
            self.msFilter.saveFile(path + "filter" + extension)

        # Save meanshift segment
        if self.msSegment is not None:
            self.msSegment.saveFile(path + "segment" + extension)

    # ------------------------------------------------------------------------
    # Meanshift functions
    # ------------------------------------------------------------------------

    def meanShiftFilter(self, hs, hr, k, eps):

        '''This function should apply a meanshift filter throughout the image.
        It takes in several arguments, including softening coefficients hs and
        hr, respectively for regular space and color space; k, the number of
        nearest neighbors to look at when filtering a local kernel; and eps,
        the threshold used to stop the iterating.'''

        # Make a copy of original image
        img = self.original.copy()

        # Initialize kernel matrix
        kernels = np.empty((img.height, img.width, 5), dtype="float64")

        # Fill first dimension with kernel's Y coordinate value
        for j in np.arange(img.height):
            kernels[j, :, 0] = j

        # Fill second dimension with kernel's X coordinate value
        for i in np.arange(img.width):
            kernels[:, i, 1] = i

        # Fill remaining dimensions with LUV values for each pixel
        kernels[:, :, 2:] = img.copy().convertRGBtoLUV().pixels

        # Initialize variables
        checks = np.ones((img.height, img.width))
        size = img.height * img.width

        # Wait until all vectors' magnitudes go below threshold
        while True in checks:

            # Iniitalize vectors for this iteration
            vectors = np.empty((img.height, img.width, 5), dtype="float64")

            # Initialize kdtree with linearized matrix and optimized for space
            kdTree = sp.spatial.cKDTree(kernels.reshape((size, 5)),
                                        compact_nodes=True,
                                        balanced_tree=True)

            # Iterate every kernel
            for j in np.arange(img.height):
                for i in np.arange(img.width):

                    # Initialize variables
                    mainSum = np.zeros((5), dtype="float64")
                    weightSum = 0.0
                    base = kernels[j][i]

                    # Query kdtree for the k nearest neighbors
                    dist, ind = kdTree.query(base, k+1, 0.01, n_jobs=-1)

                    # Iterate each pair distance, index found on the query
                    for d, i in dist[1:], ind[1:]:

                        # Get 2D indexes from 1D one
                        y = i/img.height
                        x = i - y*img.height
                        data = kernels[y][x]

                        # Compute spatial component
                        weightS = data[:2] - base[:2]
                        weightS /= (1.0 * hs)
                        weightS = -sum(weightS ** 2)
                        weightS = np.exp(weightS)

                        # Compute color component
                        weightR = data[2:] - base[2:]
                        weightR /= (1.0 * hr)
                        weightR = -sum(weightR ** 2)
                        weightR = np.exp(weightR)

                        # Resulting weight is product
                        weight = weightS * weightR

                        # Add things to variables
                        weightSum += weight
                        mainSum += data * weight

                    # Once done, result is average
                    result = mainSum / weightSum
                    vectors[j][i] = result

                    # Check if magnitude is under eps
                    checks[j][i] = np.linalg.norm(result)

            # Add vectors to kernels
            kernels += vectors

            # Checks if another iteration will be necessary
            checks <= eps

        # Once done, copy the original pixel matrix
        kernels = kernels.astype("uint64")
        colors = img.pixels.copy()

        # Iterate every pixel, changing its color from the one given by its
        # kernel after iterating. Rounds down no matter the decimal part
        for j in np.arange(img.height):
            for i in np.arange(img.width):
                y, x = kernels[j][i][:2]
                img.pixels[j][i] = colors[y][x]

        # Store result in class variable
        self.msFilter = img
