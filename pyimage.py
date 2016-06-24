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

        '''This function should implement a generic binary operation between
        two instances of PyImage. The arguments taken are itself, another
        instance, and the function to be applied to the pixel matrix.'''

        # Checks if images have same number and type of channels
        if self.img.mode != other.img.mode:
            print "\nERROR: Images must have same number of channels\n"
            return

        # Checks if images have same dimensions
        if (self.width != other.width or self.height != other.height):
            print "\nERROR: Images must have same width and height\n"
            return

        # Create third instance of PyImage, will be our result
        res = PyImage()

        # We must cast the pixels matrix to floating point from 8bit ints,
        # to prevent overflow and underflow when operating with pixels
        res.pixels = func(self.pixels.astype("float64"),
                          other.pixels.astype("float64"))

        # After applying the function, we saturate values above 255 to said
        # value, and truncate values below 0 similarly. We then cast the array
        # back into 8bit integer format
        res.pixels = np.clip(res.pixels, 0, 255).astype("uint8")

        # Once done, we build an image from the pixels matrix and return
        res.img = Image.fromarray(res.pixels, self.img.mode)
        res.width = res.img.size[0]
        res.height = res.img.size[1]

        return res

    def __add__(self, other):

        '''This function should implement basic addition of two PyImage
        instances. It uses the generic binary operator.'''

        return self._operate(other, np.add)

    def __sub__(self, other):

        '''This function should implement basic subtraction of two PyImage
        instances. It uses the generic binary operator.'''

        return self._operate(other, np.subtract)

    def __mul__(self, other):

        '''This function should implement basic multiplication of two PyImage
        instances. It uses the generic binary operator.'''

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

        '''This function should update the stored image based on the current
        state of the pixels matrix.'''

        self.img = Image.fromarray(self.pixels, self.img.mode)
        self.width = self.img.size[0]
        self.height = self.img.size[1]

    def updatePixels(self):

        '''This function should update the pixels matrix based on the current
        image saved in this instance.'''

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

        '''This function should return a deep copy of the current image and
        pixel matrix.'''

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
    # Converters
    # ------------------------------------------------------------------------

    def matrixConvert(self, matrix):

        '''This function should...'''

        for j in np.arange(self.height):
            for i in np.arange(self.width):
                px = self.pixels[j][i]
                x = sum(matrix[0]*px[0])
                y = sum(matrix[1]*px[1])
                z = sum(matrix[2]*px[2])
                self.pixels[j][i] = np.array([x, y, z])

    def convertRGBtoXYZ(self):

        '''This function should...'''

        matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])

        self.matrixConvert(matrix)

    def convertXYZtoLUV(self):

        '''This function should...'''

        whiteRefY = 1
        whiteRefu = 0.19784977571475
        whiteRefv = 0.46834507665248
        eps = (6/29.0)**3
        const = (29/3.0)**3
        power = 1/3.0

        for j in np.arange(self.height):
            for i in np.arange(self.width):
                px = self.pixels[j][i]
                y = px[1]/whiteRefY

                if y > eps:
                    L = 116*(y**(power)) - 16
                else:
                    L = const*y

                magic = px[0] + 15*px[1] + 3*px[2]
                if magic == 0:
                    up = 4
                    vp = 9/15.0
                else:
                    up = 4*px[0] / magic
                    vp = 9*px[1] / magic

                u = 13 * L * (up - whiteRefu)
                v = 13 * L * (vp - whiteRefv)

                self.pixels[j][i] = np.array([L, u, v])

    def convertRGBtoLUV(self):

        '''This function should...'''

        self.convertRGBtoXYZ()
        self.convertXYZtoLUV()

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
        # 1/(size**2) so that when function sum is applied, we compute the mean
        length = size * 2 + 1
        if self.pixels.ndim < 3:
            self.filter(size, np.ones((length, length)) / (length**2), np.sum)

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

        '''This function should implement a sobel filter, using the generic
        filter function above. Currently it is hardocded for a window of size
        3x3 only, and we build the weights matrix manually. Once done, we apply
        the derivative filter on both axis and compute the final image.'''

        # Hardcoded implementation of X and Y derivative filters
        if self.pixels.ndim < 3:
            weights_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            weights_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Hardcoded implementation of X and Y derivative filters, adapted for
        # a generic number of channels
        else:
            dim = len(self.pixels[0][0])
            weights_x = np.array([[[-1]*dim, [0]*dim, [1]*dim],
                                 [[-2]*dim, [0]*dim, [2]*dim],
                                 [[-1]*dim, [0]*dim, [1]*dim]])
            weights_y = np.array([[[-1]*dim, [-2]*dim, [-1]*dim],
                                 [[0]*dim, [0]*dim, [0]*dim],
                                 [[1]*dim, [2]*dim, [1]*dim]])

        # Create a copy of this instance so we can apply X and Y derivatives
        # separately
        copy = self.copy()

        # Here, we cast the matrix to 64bit integers, so we can go beyond the
        # 0-255 range. After operating, we need to cast it back to 8bit so the
        # image library recognizes the numbers properly
        self.pixels = self.pixels.astype("int64")
        self.filter(1, weights_x, np.sum)
        copy.pixels = copy.pixels.astype("int64")
        copy.filter(1, weights_y, np.sum)
        self.pixels = np.sqrt(self.pixels**2 + copy.pixels**2).astype("uint8")

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

        '''This function should expand an image so that both of its dimensions
        are doubled. After expanding the dimensions and filling new pixels with
        zeros, we apply a specific blue filter on the image. It should be noted
        that during pyramid operations, pixel rows and columns might be lost
        due to integer division, so we add the possibility of adding back those
        with the optional "loss" argument, that states whether we need to
        restore information to the (Y,X) axis. This is used in pyramid
        operations.'''

        # Keep copy of current pixel matrix
        old_pixels = self.pixels.copy()

        # Create new empty pixel matrix, with dimensions twice as large as
        # before. If needed, we add an extra row or column according to the
        # loss argument.
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

        # Cast new pixels matrix to unsigned 8bit integers
        self.pixels = self.pixels.astype("uint8")

        # Insert the old pixel matrix in the new one, alternating rows and
        # columns. We have to be careful since the assignment requires the
        # matrices to have same shape, so we add a -1 to limit a possible extra
        # row or column added by loss reconstruction
        self.pixels[:-1:2, :-1:2] = old_pixels

        self.updateImage()

        # Blur filter we should apply to the image
        arr = np.array([1, 4, 6, 4, 1]) / 16.0

        # To create a proper 5x5 filter, we create the weights matrix, taking
        # into account the number of channels in the image
        if self.pixels.ndim < 3:
            weights = np.empty((5, 5))
        else:
            weights = np.empty((5, 5, len(self.pixels[0][0])))

        # Then, we convolve the 1D filter with itself, creating a proper 5x5
        # filter we can use with our generic filter function
        for i in range(5):
            for j in range(5):
                if self.pixels.ndim < 3:
                    weights[i][j] = arr[i] * arr[j]
                else:
                    weights[i][j] = (arr[i] * arr[j],) * len(self.pixels[0][0])

        self.filter(2, weights, np.sum)

        # We multiply the final result by 4 since for every pixel in the new
        # image, we created 3 new black ones, and we "smudged" the original[j]
        # pixels' intensity over the new ones, effectively dividing the average
        # intensity by 4, so we multiply the image by 4 so we dont end up
        # darkening it as we expand it
        self.pixels *= 4

    def blend(self, other, mask):

        '''This function should blend two images, following a given mask. The
        arguments are itself, another instance of PyImage, and a mask that will
        be used to give weights to each image's intensity. It is very important
        that the mask has the same intensity across all channels.'''

        # Check if images have same number and type of channel
        if self.img.mode != other.img.mode:
            print "\nERROR: Images must have same number of channels\n"
            return

        # Check if images and mask all have same dimensions
        if (self.width != other.width or self.height != other.height or
                self.width != mask.width or self.height != mask.height):
            print "\nERROR: Images must have same width and height\n"
            return

        # Cast mask into interval 0-1
        mask_p = mask.pixels / 255.0

        # Extract pixel information from images
        img_a = self.pixels
        img_b = other.pixels

        # Blend the two images and cast the pixel matrix back to 8bit integer
        pix = (mask_p * img_a) + ((1 - mask_p) * img_b)
        pix = pix.astype("uint8")

        # Store result in new instance and return it
        res = PyImage()
        res.loadImage(Image.fromarray(pix, self.img.mode))
        return res

    def meanShiftFilter(self, hs, hr):

        '''This function should... '''

        kernels = np.empty((self.height, self.width, 5))

        for j in np.arange(self.height):
            kernels[j, :, 0] = j

        for i in np.arange(self.width):
            kernels[:, i, 1] = i

        kernels[:, :, 2:] = self.copy().convertRGBtoLUV().pixels
