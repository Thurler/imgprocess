from PIL import Image
from pyimage import PyImage
from gausspyramid import GaussPyramid

import numpy as np


class LaplacePyramid(object):

    '''This class represents a laplace pyramid of a given image. It contains
    an array of PyImage instances, where the higher the index, the deeper you
    are in the pyramid; it also contains a gaussian pyramid that, when paired
    with the laplacian pyramid, allows collapsing onto the original image.'''

    def __init__(self, reduce_default=4):

        self.pyramid = []
        self.gauss_pyramid = GaussPyramid(reduce_default)
        self.reduce_default = reduce_default

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFile(self, filepath):

        '''This function should initialize the class by loading a guassian
        pyramid from the image from the path specified. It then reduces that
        pyramid as much as possible.'''

        # Reset pyramid
        if not self.pyramid:
            self.pyramid = []

        # Load gauss pyramid and reduce it
        self.gauss_pyramid.loadFile(filepath)
        self.gauss_pyramid.reduceMax()

    def loadImage(self, image):

        '''This function should initialize the class by loading a guassian
        pyramid from the image specified. It then reduces that pyramid as much
        as possible.'''

        # Reset pyramid
        if not self.pyramid:
            self.pyramid = []

        # Load gauss pyramid and reduce it
        self.gauss_pyramid.loadImage(image)
        self.gauss_pyramid.reduceMax()

    def savePyramid(self, filepath):

        '''This function should save the images in the pyramid into different
        files, following the filename given in the function call. Images are
        saved in format "<name>-<level>.<extension>", where <name> and
        <extension> are given with filepath, and <level> specifies which level
        is being saved.'''

        # Level counter, starts at lowest level [0]
        count = 0

        # Separate filename from extension
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        # Save each level separately
        for image in self.pyramid:
            image.saveFile(path + str(count) + extension)
            count += 1

        # Save the gaussian pyramid as well
        self.gauss_pyramid.savePyramid(path+'gpyramid'+extension)

    # ------------------------------------------------------------------------
    # Pyramid operations
    # ------------------------------------------------------------------------

    def buildPyramid(self):

        '''This function should build the pyramid based on the gaussian pyramid
        currently stored in the class. The procedure is, for every level except
        the last one, we take level i and subtract it from the expanded level
        i - 1.'''

        # Reset pyramid
        self.pyramid = []

        # For each level, subtract it from the next level's expansion
        for i in range(len(self.gauss_pyramid.pyramid) - 1):
            self.pyramid.append(self.gauss_pyramid.pyramid[i] -
                                self.gauss_pyramid.expand(i+1))

    def collapsePyramid(self, filepath, loss=None):

        '''This function should collapse the pyramid, based on the image stored
        at the highest level in the gaussian pyramid. We expand that and add
        the result to the highest level in the laplace pyramid, and repeat the
        process for every level below, until we reach the base. We save every
        intermediate result.'''

        # Initialize result as gaussian pyramid's highest level
        result = self.gauss_pyramid.pyramid[-1].copy()

        # Count for saving images, we start at the highest level minus one
        count = len(self.pyramid) - 1

        # Separate filename from extension
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        for level in self.pyramid[::-1]:
            if loss is not None:
                # If loss array is specified, read loss from there
                result.expand(loss[count+1])
            else:
                # Else, read it from gaussian pyramid
                result.expand(self.gauss_pyramid.info_loss[count+1])
            # Add expansion to current laplacian pyramid level
            result = level + result
            # Save file to disk
            result.saveFile(path + str(count) + extension)
            count -= 1

    def blend(self, other, mask, gauss):

        '''This function should blend two laplacian pyramids, self and other,
        using a mask gaussian pyramid as reference. The function also takes in
        an image to start the gauss pyramid of the new laplace pyramid, since
        that cannot be empty. This function returns a new laplace pyramid,
        which levels are the blending of each pyramid's levels.'''

        # Check if all pyramids exist
        if not self.pyramid or not other.pyramid or not mask.pyramid:
            print "\nERROR: Please make sure all the pyramids are built\n"
            return

        # Check if pyramids have same number of levels
        if (len(self.pyramid) != len(other.pyramid) or
                len(self.pyramid) != len(mask.pyramid) - 1):
            print "\nERROR: Please make sure the pyramids have the same size\n"
            return

        # Create new lapalce pyramid and store the given image at the new gauss
        # pyramid's lowest level
        res = LaplacePyramid(self.reduce_default)
        res.gauss_pyramid.pyramid.append(gauss)

        # For every level, the result pyramid is the blending of each pyramid's
        # level
        for i in range(len(self.pyramid)):
            res.pyramid.append(self.pyramid[i].blend(other.pyramid[i],
                                                     mask.pyramid[i]))

        return res
