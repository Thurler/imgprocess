from PIL import Image
from pyimage import PyImage

import numpy as np


class GaussPyramid(object):

    '''This class...'''

    def __init__(self):

        self.pyramid = []
        self.info_loss = []

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFile(self, filepath):

        '''This function should...'''

        if not self.pyramid:
            self.pyramid = []

        self.img = PyImage()
        self.img.loadFile(filepath)
        self.pyramid.append(self.img)
        self.info_loss.append((False, False))

    def loadImage(self, image):

        '''This function should...'''

        if not self.pyramid:
            self.pyramid = []

        self.img = PyImage()
        self.img.loadImage(image)
        self.pyramid.append(self.img)
        self.info_loss.append((False, False))

    def savePyramid(self, filepath):

        '''This function should...'''

        count = 0
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        for image in self.pyramid:
            image.saveFile(path + str(count) + extension)
            count += 1

    # ------------------------------------------------------------------------
    # Pyramid operations
    # ------------------------------------------------------------------------

    def reduceMax(self):

        '''This function should...'''

        size = self.pyramid[-1].img.size

        for i in range(4):
            self.reduce()
            size = self.pyramid[-1].img.size

    def reduce(self):

        '''This function should...'''

        if not self.pyramid:
            print "\nERROR: Please load an image first\n"
            return

        img = self.pyramid[-1].copy()

        arr = np.array([1, 4, 6, 4, 1]) / 16.0

        if img.pixels.ndim < 3:
            weights = np.empty((5, 5))
        else:
            weights = np.empty((5, 5, len(img.pixels[0][0])))

        for i in range(5):
            for j in range(5):
                if img.pixels.ndim < 3:
                    weights[i][j] = arr[i] * arr[j]
                else:
                    weights[i][j] = (arr[i] * arr[j],) * len(img.pixels[0][0])

        img.filter(2, weights, np.sum)

        loss = []
        if len(img.pixels) % 2:
            loss.append(True)
        else:
            loss.append(False)

        if len(img.pixels[0]) % 2:
            loss.append(True)
        else:
            loss.append(False)

        img.pixels = img.pixels[:-1:2, :-1:2]
        img.updateImage()

        self.pyramid.append(img)
        self.info_loss.append(loss)

    def expand(self, level):

        '''This function should...'''

        if not self.pyramid:
            print "\nERROR: Please load an image first\n"
            return

        if level < 0:
            print "\nERROR: Please use non-negative index values\n"
            return

        try:
            img = self.pyramid[level].copy()
            loss = self.info_loss[level]

        except IndexError:
            print "\nERROR: Please specify a valid index\n"
            return

        img.expand(loss)
        return img
