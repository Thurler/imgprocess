from PIL import Image

import numpy as np


class PyImage(object):

    '''This class represents a single instance of Image class. It implements
    other attributes and methods that are built on top of Image's basic calls
    to develop several functions from scratch.'''

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadImage(self, image):

        '''This function should...'''

        if isinstance(image, Image.Image):
            self.img = image
            self.pixels = np.array(image.getdata())
            self.width = self.img.size[0]
            self.height = self.img.size[1]
        else:
            print "\nERROR: Argument is not an instance of Image class\n"

    def loadFile(self, filepath):

        '''This function should...'''

        try:
            self.img = Image.open(filepath)
            self.pixels = np.array(self.img.getdata())
            self.width = self.img.size[0]
            self.height = self.img.size[1]
        except IOError:
            print "\nERROR: File not found.\n"

    def saveFile(self, filepath):

        '''This function should...'''

        try:
            self.img.putdata(self.pixels)
            self.img.save(filepath)
        except IOError:
            print "\nERROR: File could not be saved.\n"

    # ------------------------------------------------------------------------
    # Channel splitting functions
    # ------------------------------------------------------------------------

    def redChannel(self):

        '''This function should...'''

        if self.img.mode not in ['RGB', 'RGBA']:
            print "\nERROR: Image does not have a red channel\n"
            return

        channel = PyImage()
        channel.loadImage(self.img.split()[0])
        return channel

    def greenChannel(self):

        '''This function should...'''

        if self.img.mode not in ['RGB', 'RGBA']:
            print "\nERROR: Image does not have a green channel\n"
            return

        channel = PyImage()
        channel.loadImage(self.img.split()[1])
        return channel

    def blueChannel(self):

        '''This function should...'''

        if self.img.mode not in ['RGB', 'RGBA']:
            print "\nERROR: Image does not have a blue channel\n"
            return

        channel = PyImage()
        channel.loadImage(self.img.split()[2])
        return channel

    # ------------------------------------------------------------------------
    # Pixel reading and writing
    # ------------------------------------------------------------------------

    def getPixel(self, x, y):

        '''This function should...'''

        if (x < 0 or x >= self.width - 1):
            print "\nERROR: Pixel X coordinate out of range\n"
            return

        if (y < 0 or y >= self.height - 1):
            print "\nERROR: Pixel Y coordinate out of range\n"
            return

        index = x * self.width + y

        return self.pixels[index]

    def setPixel(self, x, y, value):

        '''This functions should...'''

        if (x < 0 or x >= self.width - 1):
            print "\nERROR: Pixel X coordinate out of range\n"
            return

        if (y < 0 or y >= self.height - 1):
            print "\nERROR: Pixel Y coordinate out of range\n"
            return

        index = x * self.width + y

        pixels[index] = value

    # ------------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------------

    def convertBinary(self, threshold):

        '''This function should...'''

        self.pixels /= threshold
        self.pixels /= self.pixels
        self.pixels *= 255

    def negate(self):

        '''This function should...'''

        self.pixels = 255 - self.pixels

    def gammaCorrection(self, gamma):

        '''This function should...'''

        if gamma <= 0:
            print "\nERROR: Gamma argument cannot be zero or lower"
            return

        power = 1.0 / gamma

        self.pixels = (255 * ((self.pixels / 255) ** power)) // 1
