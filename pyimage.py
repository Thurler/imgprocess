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

    def getPixel(self, x, y):

        '''This function should...'''

        if (x < 0 or x > self.width - 1):
            print "\nERROR: Pixel X coordinate out of range\n"
            return

        if (y < 0 or y > self.height - 1):
            print "\nERROR: Pixel Y coordinate out of range\n"
            return

        return self.pixels[y][x]

    def setPixel(self, x, y, value):

        '''This function should...'''

        if (x < 0 or x > self.width - 1):
            print "\nERROR: Pixel X coordinate out of range\n"
            return

        if (y < 0 or y > self.height - 1):
            print "\nERROR: Pixel Y coordinate out of range\n"
            return

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
