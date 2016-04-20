from PIL import Image
from pyimage import PyImage
from gausspyramid import GaussPyramid

import numpy as np


class LaplacePyramid(object):

    '''This class...'''

    def __init__(self):

        self.pyramid = []
        self.gauss_pyramid = GaussPyramid()

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFile(self, filepath):

        '''This function should...'''

        if not self.pyramid:
            self.pyramid = []

        self.gauss_pyramid.loadFile(filepath)
        self.gauss_pyramid.reduceMax()

    def loadImage(self, image):

        '''This function should...'''

        if not self.pyramid:
            self.pyramid = []

        self.gauss_pyramid.loadImage(image)
        self.gauss_pyramid.reduceMax()

    def savePyramid(self, filepath):

        '''This function should...'''

        count = 0
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        for image in self.pyramid:
            image.saveFile(path + str(count) + extension)
            count += 1

        self.gauss_pyramid.savePyramid(path+'gpyramid'+extension)

    # ------------------------------------------------------------------------
    # Pyramid operations
    # ------------------------------------------------------------------------

    def buildPyramid(self):

        '''This function should...'''

        self.pyramid = []

        for i in range(len(self.gauss_pyramid.pyramid) - 1):
            self.pyramid.append(self.gauss_pyramid.pyramid[i] -
                                self.gauss_pyramid.expand(i+1))

    def collapsePyramid(self, filepath):

        '''This function should...'''

        result = self.gauss_pyramid.pyramid[-1].copy()

        count = len(self.pyramid) - 1
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        for level in self.pyramid[::-1]:
            result.expand()
            result = level + result
            result.saveFile(path + str(count) + extension)
            count -= 1

    def blend(self, other, mask, gauss):

        '''This function should...'''

        res = LaplacePyramid()
        res.gauss_pyramid.pyramid.append(gauss)

        pass
