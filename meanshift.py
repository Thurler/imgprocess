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

        '''This function should...'''

        self.msFilter = None
        self.msSegment = None

        self.original = PyImage()
        self.original.loadFile(filepath)

    def loadImage(self, image):

        '''This function should...'''

        self.msFilter = None
        self.msSegment = None

        # Create image, append things
        self.original = PyImage()
        self.original.loadImage(image)

    def savePyramid(self, filepath):

        '''This function should...'''

        # Separate name from extension
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        self.original.saveFile(path + "original" + extension)

        if self.msFilter is not None:
            self.msFilter.saveFile(path + "filter" + extension)

        if self.msSegment is not None:
            self.msSegment.saveFile(path + "segment" + extension)

    # ------------------------------------------------------------------------
    # Meanshift functions
    # ------------------------------------------------------------------------

    def meanShiftFilter(self, hs, hr, eps, k):

        '''This function should... '''

        img = self.original.copy()

        kernels = np.empty((img.height, img.width, 5), dtype="float64")

        for j in np.arange(img.height):
            kernels[j, :, 0] = j

        for i in np.arange(img.width):
            kernels[:, i, 1] = i

        kernels[:, :, 2:] = img.copy().convertRGBtoLUV().pixels

        vectors = [1]
        size = img.height * img.width

        while True in vectors:

            vectors = np.empty((img.height, img.width, 5), dtype="float64")
            kdTree = sp.spatial.cKDTree(kernels.reshape((size, 5)),
                                        compact_nodes=True,
                                        balanced_tree=True)

            for j in np.arange(img.height):
                for i in np.arange(img.width):

                    mainSum = np.zeros((5), dtype="float64")
                    weightSum = 0.0
                    base = kernels[j][i]

                    dist, ind = kdTree.query(base, k+1, 0.01, n_jobs=-1)

                    for d, i in dist[1:], ind[1:]:
                        y = i/img.height
                        x = i - y*img.height
                        data = kernels[y][x]

                        weightS = data[:2] - base[:2]
                        weightS /= (1.0 * hs)
                        weightS = -sum(weightS ** 2)
                        weightS = np.exp(weightS)

                        weightR = data[2:] - base[2:]
                        weightR /= (1.0 * hr)
                        weightR = -sum(weightR ** 2)
                        weightR = np.exp(weightR)

                        weight = weightS * weightR

                        weightSum += weight
                        mainSum += data * weight

                    vectors[j][i] = mainSum / weightSum

            kernels += vectors
            vectors <= eps

        kernels = kernels.astype("uint64")
        colors = img.pixels.copy()

        for j in np.arange(img.height):
            for i in np.arange(img.width):
                y, x = kernels[j][i][:2]
                img.pixels[j][i] = colors[y][x]

        self.msFilter = img
