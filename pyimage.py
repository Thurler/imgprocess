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
        else:
            print "\nERROR: Argument is not an instance of Image class\n"

    def loadFile(self, filepath):

        '''This function should...'''

        try:
            self.img = Image.open(filepath)
        except IOError:
            print "\nERROR: File not found.\n"

    def saveFile(self, filepath):

        '''This function should...'''

        try:
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

        channel = PyImage()
        channel.loadImage(self.img.split()[0])
        return channel

    def greenChannel(self):

        '''This function should...'''

        if self.img.mode not in ['RGB', 'RGBA']:
            print "\nERROR: Image does not have a green channel\n"

        channel = PyImage()
        channel.loadImage(self.img.split()[1])
        return channel

    def blueChannel(self):

        '''This function should...'''

        if self.img.mode not in ['RGB', 'RGBA']:
            print "\nERROR: Image does not have a blue channel\n"

        channel = PyImage()
        channel.loadImage(self.img.split()[2])
        return channel

    # ------------------------------------------------------------------------
    # Channel converting functions
    # ------------------------------------------------------------------------

    def convertBinary(self):

        '''This function should...'''

        if self.img.mode is '1':
            print "\nERROR: Image is already a binary image.\n"
            return

        self.img.convert('1')

    def convertGrayscale(self):

        '''This function should...'''

        if self.img.mode is 'L':
            print "\nERROR: Image is already a grayscale image.\n"
            return

        self.img.convert('L')

    # ------------------------------------------------------------------------
    # Pixel reading and writing
    # ------------------------------------------------------------------------

    def getPixel(x, y):

        '''This function should...'''

        pass

    def setPixel(x, y):

        '''This functions should...'''

        pass
