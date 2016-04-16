from PIL import Image

import numpy as np


class PyImage(object):

    '''This class represents a single instance of Image class. It implements
    other attributes and methods that are built on top of Image's basic calls
    to develop several functions from scratch.'''

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def init(self):

        '''This function should...'''

        self.pixels = np.array(self.img)

        self.width = self.img.size[0]
        self.height = self.img.size[1]

    def loadImage(self, image):

        '''This function should...'''

        if isinstance(image, Image.Image):
            self.img = image
            self.init()

        else:
            print "\nERROR: Argument is not an instance of Image class\n"

    def loadFile(self, filepath):

        '''This function should...'''

        try:
            self.img = Image.open(filepath)
            self.init()

        except IOError:
            print "\nERROR: File not found.\n"

    def saveFile(self, filepath):

        '''This function should...'''

        try:
            Image.fromarray(self.pixels, self.img.mode).save(filepath)

        except IOError:
            print "\nERROR: File could not be saved.\n"

    # ------------------------------------------------------------------------
    # Pixel reading and writing
    # ------------------------------------------------------------------------

    def getPixel(self, x, y, c=None):

        '''This function should...'''

        if (x < 0 or x > self.width - 1):
            print "\nERROR: Pixel X coordinate out of range\n"
            return

        if (y < 0 or y > self.height - 1):
            print "\nERROR: Pixel Y coordinate out of range\n"
            return

        if c is not None:
            return self.pixels[y][x][c]
        else:
            return self.pixels[y][x]

    def setPixel(self, value, x, y, c=None):

        '''This function should...'''

        if (x < 0 or x > self.width - 1):
            print "\nERROR: Pixel X coordinate out of range\n"
            return

        if (y < 0 or y > self.height - 1):
            print "\nERROR: Pixel Y coordinate out of range\n"
            return

        if c is not None:
            self.pixels[y][x][c] = value
        else:
            self.pixels[y][x] = value

    # ------------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------------

    def negate(self):

        '''This function should...'''

        self.pixels = 255 - self.pixels

    def gammaCorrection(self, gamma):

        '''This function should...'''

        if gamma <= 0:
            print "\nERROR: Gamma argument cannot be zero or lower"
            return

        power = 1.0 / gamma

        self.pixels = (((self.pixels / 255.0) ** power) * 255).astype("uint8")

    def filter(self, size, weights, func):

        '''This function should...'''

        if self.pixels.ndim < 3:
            window = np.empty((size*2+1, size*2+1), dtype=np.ndarray)
        else:
            window = np.empty((size*2+1, size*2+1, len(self.pixels[0][0])),
                              dtype=np.ndarray)

        if weights.shape != window.shape:
            print "\nERROR: Weights matrix needs to be same size as window\n"
            return

        for j in np.arange(size, self.height-size):
            for i in np.arange(size, self.width-size):

                img_slice = self.pixels[j-size:j+size+1, i-size:i+size+1]
                window = weights * img_slice

                if self.pixels.ndim < 3:
                    self.setPixel(func(window), i, j)
                    continue

                for c in range(len(self.pixels[0][0])):
                    self.setPixel(func(window[:, :, c]), i, j, c)

    def blur(self, size):

        '''This function should...'''

        if not isinstance(size, int) or size < 1:
            print "\nERROR: Blur size needs to be an integer higher than 0"
            return

        if self.pixels.ndim < 3:
            self.filter(size, np.ones((size*2+1, size*2+1)), np.mean)

        else:
            self.filter(size,
                        np.ones((size*2+1, size*2+1, len(self.pixels[0][0]))),
                        np.mean)

    def medianFilter(self, size):

        '''This function should...'''

        if not isinstance(size, int) or size < 1:
            print "\nERROR: Blur size needs to be an integer higher than 0"
            return

        if self.pixels.ndim < 3:
            self.filter(size, np.ones((size*2+1, size*2+1)), np.median)

        else:
            self.filter(size,
                        np.ones((size*2+1, size*2+1, len(self.pixels[0][0]))),
                        np.median)

    # ------------------------------------------------------------------------
    # Data Processing
    # ------------------------------------------------------------------------

    def histogram(self):

        '''This function should...'''

        if self.pixels.ndim < 3:
            return np.histogram(self.pixels, np.arange(256))[0]

        else:
            res = []
            for i in range(len(self.pixels[0][0])):
                res.append(np.histogram(self.pixels[:, :, i],
                                        np.arange(256))[0])
            return tuple(res)
