from PIL import Image

import numpy as np


class PyImage(object):

    '''This class represents a single instance of Image class. It implements
    other attributes and methods that are built on top of Image's basic calls
    to develop several functions from scratch.'''

    def __init__(self):

        self.img = None
        self.pixels = None
        self.width = 0
        self.height = 0

    # ------------------------------------------------------------------------
    # Interclass operators
    # ------------------------------------------------------------------------

    def _operate(self, other, func):

        '''This function should...'''

        print self.width, other.width, self.height, other.height

        if self.img.mode != other.img.mode:
            print "\nERROR: Images must have same number of channels\n"
            return

        if (self.width != other.width or self.height != other.height):
            print "\nERROR: Images must have same width and height\n"
            return

        res = PyImage()
        res.pixels = func(self.pixels.astype("int64"),
                          other.pixels.astype("int64"))
        res.pixels = np.clip(res.pixels, 0, 255).astype("uint8")
        res.img = Image.fromarray(res.pixels, self.img.mode)
        res.width = res.img.size[0]
        res.height = res.img.size[1]

        return res

    def __add__(self, other):

        '''This function should...'''

        return self._operate(other, np.add)

    def __sub__(self, other):

        '''This function should...'''

        return self._operate(other, np.subtract)

    def __mul__(self, other):

        '''This function should...'''

        return self._operate(other, np.multiply)

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def init(self):

        '''This function should initialize the width and height variables,
        as well as load pixels into a numpy 2D array (3D if image is RGB).'''

        # Read pixels in 3D matrix format (Y, X, C), where C is color channel
        self.pixels = np.array(self.img)

        # Set width and height for later use
        self.width = self.img.size[0]
        self.height = self.img.size[1]

    def updateImage(self):

        '''This function should...'''

        self.img = Image.fromarray(self.pixels, self.img.mode)
        self.width = self.img.size[0]
        self.height = self.img.size[1]

    def updatePixels(self):

        '''This function should...'''

        self.pixels = np.array(self.img)

    def loadImage(self, image):

        '''This function should initialize the class with a given Image object.
        It is also used when copying PyImage instances, since it loads a new
        matrix of pixels in memory.'''

        # Checks if argument is an Image, initializes class
        if isinstance(image, Image.Image):
            self.img = image
            self.init()

        else:
            print "\nERROR: Argument is not an instance of Image class\n"

    def loadFile(self, filepath):

        '''This function should initialize the class with an image, given a
        file path. If the file doesn't exist, ir errors out.'''

        try:
            # Reads file from filepath and opens it as a image
            self.img = Image.open(filepath)
            self.init()

        except IOError:
            print "\nERROR: File not found.\n"

    def saveFile(self, filepath):

        '''This function should output the current state of pixels matrix into
        a file given by filepath argument.'''

        try:
            # Save current pixels matrix to a file
            Image.fromarray(self.pixels, self.img.mode).save(filepath)

        except IOError:
            print "\nERROR: File could not be saved.\n"

    def copy(self):

        '''This function should return a copy of the current image and pixel
        matrix.'''

        pyimg = PyImage()
        pyimg.loadImage(self.img.copy())
        return pyimg

    # ------------------------------------------------------------------------
    # Pixel reading and writing
    # ------------------------------------------------------------------------

    def getPixel(self, x, y, c=None):

        '''This function should return the intensity of a given pixel at
        coordinates (x,y). If c is specified, it returns a specific channel's
        intensity, and it is up to the user to make sure c is in range. If left
        alone, it returns a tuple of intensities for each channel.'''

        # Check for in-bound indexes
        if (x < 0 or x > self.width - 1):
            print "\nERROR: Pixel X coordinate out of range\n"
            return

        if (y < 0 or y > self.height - 1):
            print "\nERROR: Pixel Y coordinate out of range\n"
            return

        # Return specific channel
        if c is not None:
            return self.pixels[y][x][c]
        else:
            return self.pixels[y][x]

    def setPixel(self, value, x, y, c=None):

        '''This function should should set the intensity of a given pixel at
        coordinates (x,y). If c is specified, it sets a specific channel's
        intensity, and it is up to the user to make sure c is in range. If left
        alone, it sets a tuple of intensities for a pixel. It is also up to the
        user to make sure the number of elements in the tuple matches the
        number of channels.'''

        # Check for in-bound indexes
        if (x < 0 or x > self.width - 1):
            print "\nERROR: Pixel X coordinate out of range\n"
            return

        if (y < 0 or y > self.height - 1):
            print "\nERROR: Pixel Y coordinate out of range\n"
            return

        # Set specific channel
        if c is not None:
            self.pixels[y][x][c] = value
        else:
            self.pixels[y][x] = value

    # ------------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------------

    def negate(self):

        '''This function should invert every channel instensity for every pixel
        in the image.'''

        # Simply negate all the pixels in every channel
        self.pixels = 255 - self.pixels

    def gammaCorrection(self, gamma):

        '''This function should apply the gamma correction function for every
        pixel in the image. The gamma argument should be greater than zero to
        ensure the algorithm makes sense.'''

        if gamma <= 0:
            print "\nERROR: Gamma argument cannot be zero or lower"
            return

        # Take the inverse since it's gamma correction
        power = 1.0 / gamma

        # I' = 255 * (I / 255)^1/gamma
        self.pixels = (((self.pixels / 255.0) ** power) * 255).astype("uint8")

    def filter(self, size, weights, func):

        '''This function should apply a generic filter over every pixel in the
        image. The arguments are: the window size, which measures in levels of
        neighborhood to account for (a value of 1 would look at the 8 pixels
        surrounding the central pixel); the weights matrix to apply to the
        window, which should have the same dimensions as the window itself; the
        function to apply to the window to acquire the central pixel's new
        intensity value.'''

        pixels = self.pixels.copy()

        # Graycale images are represented in a 2D array, and RGB ones in 3D
        if self.pixels.ndim < 3:
            # Window is a square 2D matrix of dimensions (size*2 + 1)
            window = np.empty((size*2+1, size*2+1), dtype=np.ndarray)
        else:
            # Window is a 3D matrix of dimensions (size*2 + 1) squared and the
            # number of channels found in the image
            window = np.empty((size*2+1, size*2+1, len(self.pixels[0][0])),
                              dtype=np.ndarray)

        if weights.shape != window.shape:
            print "\nERROR: Weights matrix needs to be same size as window\n"
            return

        # Iterate every non-border pixel - generic case
        for j in np.arange(size, self.height-size):
            for i in np.arange(size, self.width-size):

                # Slice the image, centering the window at pixel (i, j)
                img_slice = pixels[j-size:j+size+1, i-size:i+size+1]

                # Apply the weights matrix to the window
                window = weights * img_slice

                if self.pixels.ndim < 3:
                    # Apply function to window
                    self.setPixel(func(window), i, j)
                    continue

                for c in range(len(self.pixels[0][0])):
                    # Apply function to window, but only for channel C
                    self.setPixel(func(window[:, :, c]), i, j, c)

        # Border treatment - upper border
        for j in range(size):
            for i in range(self.width):
                self.border_filter(pixels, i, j, size, weights, func)

        # Border treatment - lower border
        for j in range(self.height-size, self.height):
            for i in range(self.width):
                self.border_filter(pixels, i, j, size, weights, func)

        # Border treatment - left border
        for j in range(size, self.height-size):
            for i in range(size):
                self.border_filter(pixels, i, j, size, weights, func)

        # Border treatment - right border
        for j in range(size, self.height-size):
            for i in range(self.width-size, self.width):
                self.border_filter(pixels, i, j, size, weights, func)

    def border_filter(self, pixels, i, j, size, weights, func):

        '''This function applies the generic filter from the filter function
        on the image borders, and exists because they need special treatment
        for in-bound pixel indexing. This method does not wrap around and does
        not double the edge - it simply discards measurements that are out of
        bounds.'''

        elements = []

        # Given i, j, we compute the window by hand, discarding the out of
        # bounds elements:
        for y in range(-size, size+1):
            for x in range(-size, size+1):
                if i-x < 0 or i-x >= self.width:
                    # Out of bounds
                    continue
                if j-y < 0 or j-y >= self.height:
                    # Out of bounds
                    continue
                # Works for both grayscale and RGB
                element = np.array(pixels[j-y][i-x])
                elements.append(weights[size+y][size+x] * element)

        # Numpy arrays are easy to map functions to specific rows, columns or
        # pieces of the array through smart indexing
        elements = np.array(elements)

        if self.pixels.ndim < 3:
            # Apply function to window
            self.setPixel(func(elements), i, j)
            return

        for c in range(len(self.pixels[0][0])):
            # Apply function to window, but only for channel C
            self.setPixel(func(elements[:, c]), i, j, c)

    def blur(self, size):

        '''This function should blur the image, so it applies a mean filter
        over every pixel in the image. The size argument determines the window
        size, and it passes a weights matrix filled with ones to the filter
        function.'''

        if not isinstance(size, int) or size < 1:
            print "\nERROR: Blur size needs to be an integer higher than 0"
            return

        # Weights matrix is a 2D/3D (depending on channels) matrix filled with
        # ones, function is mean
        length = size * 2 + 1
        if self.pixels.ndim < 3:
            self.filter(size, np.ones((length, length)) / (length**2), np.mean)

        else:
            dim = len(self.pixels[0][0])
            self.filter(size,
                        np.ones((length, length, dim)) / (length**2),
                        np.sum)

    def medianFilter(self, size):

        '''This function should implement a median filter, that removes grainy
        noise from images. The size argument determines the window size, and it
        passes a weights matrix filled with ones to the filter function.'''

        if not isinstance(size, int) or size < 1:
            print "\nERROR: Blur size needs to be an integer higher than 0"
            return

        # Weights matrix is a 2D/3D (depending on channels) matrix filled with
        # ones, function is median
        if self.pixels.ndim < 3:
            self.filter(size, np.ones((size*2+1, size*2+1)), np.median)

        else:
            self.filter(size,
                        np.ones((size*2+1, size*2+1, len(self.pixels[0][0]))),
                        np.median)

    def sobelFilter(self):

        '''This function should...'''

        if self.pixels.ndim < 3:
            weights_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            weights_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        else:
            dim = len(self.pixels[0][0])
            weights_x = np.array([[[-1]*dim, [0]*dim, [1]*dim],
                                 [[-2]*dim, [0]*dim, [2]*dim],
                                 [[-1]*dim, [0]*dim, [1]*dim]])
            weights_y = np.array([[[-1]*dim, [-2]*dim, [-1]*dim],
                                 [[0]*dim, [0]*dim, [0]*dim],
                                 [[1]*dim, [2]*dim, [1]*dim]])

        copy = self.copy()

        self.pixels = self.pixels.astype("int64")
        self.filter(1, weights_x, np.sum)
        copy.pixels = copy.pixels.astype("int64")
        copy.filter(1, weights_y, np.sum)
        self.pixels = np.sqrt(self.pixels**2 + copy.pixels**2).astype("uint8")

    def laplacianFilter(self):

        '''This function should...'''

        if self.pixels.ndim < 3:
            weights = np.ones((3, 3))
            weights[1][1] = -8

        else:
            weights = np.ones((3, 3, 3))
            weights[1][1] = [-8]*3

        copy = self.copy()

        copy.pixels = copy.pixels.astype("int64")
        copy.filter(1, weights, np.sum)

        return copy.pixels

    # ------------------------------------------------------------------------
    # Data Processing
    # ------------------------------------------------------------------------

    def histogram(self):

        '''This function should return a histogram of intensities for each
        channel in the image.'''

        # Numpy's histogram function returns a tuple where first element is
        # distribution for each bin, and second element is the bin list. We are
        # interested in the distribution one, and the bins used is a range
        # 0-255. We do this for every channel.
        if self.pixels.ndim < 3:
            return np.histogram(self.pixels, np.arange(256))[0]

        else:
            res = []
            for i in range(len(self.pixels[0][0])):
                res.append(np.histogram(self.pixels[:, :, i],
                                        np.arange(256))[0])
            return tuple(res)

    def expand(self, loss=(False, False)):

        '''This function should...'''

        old_pixels = self.pixels.copy()
        if self.pixels.ndim < 3:
            if loss[0] and loss[1]:
                self.pixels = np.zeros((self.height*2+1, self.width*2+1))
            elif loss[0]:
                self.pixels = np.zeros((self.height*2+1, self.width*2))
            elif loss[1]:
                self.pixels = np.zeros((self.height*2, self.width*2+1))
            else:
                self.pixels = np.zeros((self.height*2, self.width*2))
        else:
            dim = len(self.pixels[0][0])
            if loss[0] and loss[1]:
                self.pixels = np.zeros((self.height*2+1, self.width*2+1, dim))
            elif loss[0]:
                self.pixels = np.zeros((self.height*2+1, self.width*2, dim))
            elif loss[1]:
                self.pixels = np.zeros((self.height*2, self.width*2+1, dim))
            else:
                self.pixels = np.zeros((self.height*2, self.width*2, dim))

        self.pixels = self.pixels.astype("uint8")
        self.pixels[:-1:2, :-1:2] = old_pixels

        self.updateImage()

        arr = np.array([1, 4, 6, 4, 1]) / 16.0

        if self.pixels.ndim < 3:
            weights = np.empty((5, 5))
        else:
            weights = np.empty((5, 5, len(self.pixels[0][0])))

        for i in range(5):
            for j in range(5):
                if self.pixels.ndim < 3:
                    weights[i][j] = arr[i] * arr[j]
                else:
                    weights[i][j] = (arr[i] * arr[j],) * len(self.pixels[0][0])

        self.filter(2, weights, np.sum)

        self.pixels *= 4

    def blend(self, other, mask):

        '''This function should...'''

        if self.img.mode != other.img.mode:
            print "\nERROR: Images must have same number of channels\n"
            return

        if (self.width != other.width or self.height != other.height or
                self.width != mask.width or self.height != mask.height):
            print "\nERROR: Images must have same width and height\n"
            return

        mask_p = mask.pixels / 255.0
        img_a = self.pixels
        img_b = other.pixels
        pix = (mask_p * img_a) + ((1 - mask_p) * img_b)
        pix = pix.astype("uint8")
        res = PyImage()
        res.loadImage(Image.fromarray(pix, self.img.mode))
        return res
