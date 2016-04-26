from PIL import Image
from pyimage import PyImage

import numpy as np


class GaussPyramid(object):

    '''This class represent a gaussian pyramid of a given image. It contains
    an array of PyImage instances, where the higher the index, the deeper you
    are in the pyramid; and also an array of loss information, so that we can
    prevent pixel lines and columns from being lost when reducing an image.'''

    def __init__(self):

        self.pyramid = []
        self.info_loss = []

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFile(self, filepath):

        '''This function should initialize the class by loading an image from
        the path specified. It then places that image at the pyramid's lowest
        level, and specifies no pixels were lost in this process.'''

        # Reset pyramid and loss info
        if self.pyramid:
            self.pyramid = []
            self.info_loss = []

        # Create image, append things
        img = PyImage()
        img.loadFile(filepath)
        self.pyramid.append(img)
        self.info_loss.append((False, False))

    def loadImage(self, image):

        '''This function should initialize the class by loading an image from
        the function call. It then places that image at the pyramid's lowest
        level, and specifies no pixels were lost in this process.'''

        # Reset pyramid and loss info
        if self.pyramid:
            self.pyramid = []
            self.info_loss = []

        # Create image, append things
        img = PyImage()
        img.loadImage(image)
        self.pyramid.append(img)
        self.info_loss.append((False, False))

    def savePyramid(self, filepath):

        '''This function should save the images in the pyramid into different
        files, following the filename given in the function call. Images are
        saved in format "<name>-<level>.<extension>", where <name> and
        <extension> are given with filepath, and <level> specifies which level
        is being saved.'''

        # Level counter, starts at lowest level [0]
        count = 0

        # Separate name from extension
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        # Save each level separately
        for image in self.pyramid:
            image.saveFile(path + str(count) + extension)
            count += 1

    # ------------------------------------------------------------------------
    # Pyramid operations
    # ------------------------------------------------------------------------

    def reduceMax(self):

        '''This function should reduce the image a fixed number of times. This
        adds a specific amount of levels to the gaussian pyramid.'''

        for i in range(4):
            self.reduce()

    def reduce(self):

        '''This function should reduce the image by a single level. This adds a
        new level to the pyramid, so the reducing is based on the current
        highest level. Before halving the image's dimensions, we blur it to
        prevent sharpening pixel intensity differences.'''

        # Check if pyramid has been generated
        if not self.pyramid:
            print "\nERROR: Please load an image first\n"
            return

        # Copy highest level's pixel matrix
        img = self.pyramid[-1].copy()

        # Blur filter we should apply to the imag
        arr = np.array([1, 4, 6, 4, 1]) / 16.0

        # To create a proper 5x5 filter, we create the weights matrix, taking
        # into account the number of channels in the image
        if img.pixels.ndim < 3:
            weights = np.empty((5, 5))
        else:
            weights = np.empty((5, 5, len(img.pixels[0][0])))

        # Then, we convolve the 1D filter with itself, creating a proper 5x5
        # filter we can use with our generic filter function
        for i in range(5):
            for j in range(5):
                if img.pixels.ndim < 3:
                    weights[i][j] = arr[i] * arr[j]
                else:
                    weights[i][j] = (arr[i] * arr[j],) * len(img.pixels[0][0])

        img.filter(2, weights, np.sum)

        # We now check if any pixels will be lost when halving the image - this
        # happens when we have an odd dimension
        loss = []
        if len(img.pixels) % 2:
            loss.append(True)
        else:
            loss.append(False)

        if len(img.pixels[0]) % 2:
            loss.append(True)
        else:
            loss.append(False)

        # Half the image, taking lines and columns alternatedly
        img.pixels = img.pixels[:-1:2, :-1:2]
        img.updateImage()

        # Append new stuff to new level
        self.pyramid.append(img)
        self.info_loss.append(loss)

    def expand(self, level):

        '''This function should expand the image stored at the specified level.
        To do so, we simply call the expand function for that image, specifying
        whether there was a loss of pixels or not in its reduction to that
        level.'''

        # Check if pyramid exists
        if not self.pyramid:
            print "\nERROR: Please load an image first\n"
            return

        # Check if index is not negative
        if level < 0:
            print "\nERROR: Please use non-negative index values\n"
            return

        # Check if index is valid
        try:
            img = self.pyramid[level].copy()
            loss = self.info_loss[level]

        except IndexError:
            print "\nERROR: Please specify a valid index\n"
            return

        # Expand image
        img.expand(loss)
        return img
