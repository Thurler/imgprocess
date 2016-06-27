from PIL import Image
from pyimage import PyImage

import threading
import itertools
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

    def kernelIteration(self, tid, hs, hr, k, kernels, newKernels,
                        kdTree, checks, ttot, img):

        '''This function is responsible for computing the new kernel position
        for kernels [id*size:(id+1)*size]'''

        for j in np.arange(tid*(img.height)/ttot, (tid+1)*(img.height)/ttot):
            for i in np.arange(img.width):
                # Initialize variables
                mainSum = np.zeros((5), dtype="float64")
                weightSum = 0.0
                base = kernels[j][i]

                # Query kdtree for the k nearest neighbors
                dist, index = kdTree.query(base, k+1, 0.01, n_jobs=-1)

                # Iterate each pair distance, index found on the query
                for d, ind in itertools.izip(dist[1:], index[1:]):

                    # Get 2D indexes from 1D one
                    y = ind/img.width
                    x = ind - y*img.width
                    data = kernels[y][x]

                    # Compute spatial component
                    weightS = data[:2] - base[:2]
                    weightS = weightS/(1.0 * hs)
                    weightS = -sum(weightS ** 2)
                    weightS = np.exp(weightS)

                    # Compute color component
                    weightR = data[2:] - base[2:]
                    weightR = weightR/(1.0 * hr)
                    weightR = -sum(weightR ** 2)
                    weightR = np.exp(weightR)

                    # Resulting weight is product
                    weight = weightS * weightR

                    # Add things to variables
                    weightSum += weight
                    mainSum += data * weight

                # Once done, result is average, if zero, kernel doesnt move
                if weightSum == 0:
                    newKernels[j][i] = base
                    checks[j][i] = 0
                    continue
                result = mainSum / weightSum
                newKernels[j][i] = result
                checks[j][i] = np.linalg.norm(result - base)

    def meanshiftFilter(self, hs, hr, k, eps, lim):

        '''This function should apply a meanshift filter throughout the image.
        It takes in several arguments, including softening coefficients hs and
        hr, respectively for regular space and color space; k, the number of
        nearest neighbors to look at when filtering a local kernel; and eps,
        the threshold used to stop the iterating.'''

        # Make copies of original image
        img = self.original.copy()
        kerns = self.original.copy()

        # Initialize kernel matrix
        kernels = np.empty((img.height, img.width, 5), dtype="float64")

        # Fill first dimension with kernel's Y coordinate value
        for j in np.arange(img.height):
            kernels[j, :, 0] = j

        # Fill second dimension with kernel's X coordinate value
        for i in np.arange(img.width):
            kernels[:, i, 1] = i

        # Fill remaining dimensions with LUV values for each pixel
        kerns.convertRGBtoLUV()
        kernels[:, :, 2:] = kerns.pixels

        # Initialize variables
        checks = np.zeros((img.height, img.width))
        boolchecks = checks.astype("bool")
        size = img.height * img.width
        kdTree = None
        newKernels = None

        # Wait until all vectors' magnitudes go below threshold
        while False in boolchecks:

            # Iniitalize vectors for this iteration
            newKernels = np.empty((img.height, img.width, 5), dtype="float64")

            # Initialize kdtree with linearized matrix and optimized for space
            kdTree = sp.spatial.cKDTree(kernels.reshape((size, 5)),
                                        compact_nodes=True,
                                        balanced_tree=True)

            # Initialize threads, iterate every kernel
            threads = []
            for i in range(4):
                t = threading.Thread(name=str(i), target=self.kernelIteration,
                                     args=(i, hs, hr, k, kernels, newKernels,
                                           kdTree, checks, 4, img))
                t.setDaemon(True)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Update kernels
            kernels = newKernels

            print "DEBUG: Checks min/avg/max:", checks.min(), \
                np.mean(checks), checks.max()

            # Checks if another iteration will be necessary
            boolchecks = checks <= eps

            stopped = np.count_nonzero(boolchecks)
            print "DEBUG: Fixed kernels:", stopped
            if stopped >= (lim*size):
                break

        # Once done, copy the original pixel matrix
        kernelsCopy = kernels.astype("uint64")
        colors = img.pixels.copy()

        # Iterate every pixel, changing its color from the one given by its
        # kernel after iterating. Rounds down no matter the decimal part
        for j in np.arange(img.height):
            for i in np.arange(img.width):
                y, x = kernelsCopy[j][i][:2]
                img.pixels[j][i] = colors[y][x]

        # Store result in class variable
        self.msFilter = img

        # Return kernels for segmentation
        return kernels
