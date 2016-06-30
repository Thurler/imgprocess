from PIL import Image
from pyimage import PyImage

import threading
import itertools
import numpy as np
import scipy as sp
from scipy import spatial


class MeanShift(object):

    '''This class should host all images related to the process of performing a
    meanshift segmentation on an image, housing functions that will perform
    such operation. It consists of a base image and two derived images, a
    filtered one and a segmented one.'''

    def __init__(self):

        self.original = None
        self.msFilter = None
        self.msSegment = None

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFile(self, filepath):

        '''This function should open the image contained in the specified file
        and reset the class' variables'''

        # Reset variables
        self.msFilter = None
        self.msSegment = None

        # Load image
        self.original = PyImage()
        self.original.loadFile(filepath)

    def loadImage(self, image):

        '''This function should open the image contained in the specified file
        and reset the class' variables'''

        # Reset variables
        self.msFilter = None
        self.msSegment = None

        # Load image
        self.original = PyImage()
        self.original.loadImage(image)

    def saveFile(self, filepath):

        '''This function should save to disk the class' current state, by
        saving not only the original file, but also the meanshifted ones, if
        they exist.'''

        # Separate name from extension
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        # Save original
        self.original.saveFile(path + "original" + extension)

        # Save meanshift filter
        if self.msFilter is not None:
            self.msFilter.saveFile(path + "filter" + extension)

        # Save meanshift segment
        if self.msSegment is not None:
            self.msSegment.saveFile(path + "segment" + extension)

    # ------------------------------------------------------------------------
    # Meanshift functions
    # ------------------------------------------------------------------------

    def kernelIteration(self, tid, hs, hr, k, kernels, newKernels, eps,
                        kdTree, checks, ttot, img, valid):

        '''This function is responsible for computing the new kernel position
        for kernels [id*size:(id+1)*size]'''

        for j in np.arange(tid*(img.height)/ttot, (tid+1)*(img.height)/ttot):
            for i in np.arange(img.width):

                # If not valid, do not iterate again
                if not valid[j][i]:
                    newKernels[j][i] = kernels[j][i]
                    checks[j][i] = 0
                    continue

                # Initialize variables
                mainSum = np.zeros((5), dtype="float64")
                weightSum = 0.0
                base = kernels[j][i]

                # Query kdtree for the k nearest neighbors
                dist, index = kdTree.query(base,
                                           k+1,
                                           distance_upper_bound=hr,
                                           n_jobs=-1)

                # Iterate each pair distance, index found on the query
                for d, ind in itertools.izip(dist[1:], index[1:]):

                    # Get 2D indexes from 1D one
                    if ind == kdTree.n:
                        continue
                    y = ind/img.width
                    x = ind - y*img.width
                    data = kernels[y][x]

                    # Compute spatial component
                    weightS = data[:2] - base[:2]
                    weightS = weightS/(1.0 * hs)
                    weightS = -sum(weightS ** 2)
                    weightS = np.exp(weightS)

                    # Compute color component
                    weightR = data[2:] - base[2:]
                    weightR = weightR/(1.0 * hr)
                    weightR = -sum(weightR ** 2)
                    weightR = np.exp(weightR)

                    # Resulting weight is product
                    weight = weightS * weightR

                    # Add things to variables
                    weightSum += weight
                    mainSum += data * weight

                # Once done, result is average, if zero, kernel doesnt move
                if weightSum == 0:
                    print i, j
                    newKernels[j][i] = kernels[j][i]
                    checks[j][i] = 0
                    valid[j][i] = False
                    continue
                result = mainSum / weightSum
                newKernels[j][i] = result
                norm = np.linalg.norm(result - base)
                checks[j][i] = norm
                if norm < eps:
                    valid[j][i] = False

    def meanshiftFilter(self, hs, hr, k, eps, lim, ops):

        '''This function should apply a meanshift filter throughout the image.
        It takes in several arguments, including softening coefficients hs and
        hr, respectively for regular space and color space; k, the number of
        nearest neighbors to look at when filtering a local kernel; and eps,
        the threshold used to stop the iterating.'''

        # Make copies of original image
        img = self.original.copy()
        kerns = self.original.copy()

        # Initialize kernel matrix
        kernels = np.empty((img.height, img.width, 5), dtype="float64")

        # Fill first dimension with kernel's Y coordinate value
        for j in np.arange(img.height):
            kernels[j, :, 0] = j

        # Fill second dimension with kernel's X coordinate value
        for i in np.arange(img.width):
            kernels[:, i, 1] = i

        # Fill remaining dimensions with LUV values for each pixel
        # kerns.convertRGBtoLUV()
        kerns.pixels = kerns.pixels.astype("float64")
        kernels[:, :, 2:] = kerns.pixels

        # Initialize variables
        checks = np.zeros((img.height, img.width))
        boolchecks = checks.astype("bool")
        size = img.height * img.width
        kdTree = None
        newKernels = None
        firstKernel = kernels.copy()

        # Initialize list of valid points for iteration
        valid = np.ones((img.height, img.width), dtype="bool")

        # Iteration count
        count = 0

        # Wait until all vectors' magnitudes go below threshold
        while False in boolchecks:

            # Interrupt if exceeps iteration limit
            count += 1
            if count > ops:
                break

            # Iniitalize vectors for this iteration
            newKernels = np.empty((img.height, img.width, 5), dtype="float64")

            # Initialize kdtree with linearized matrix and optimized for space
            kdTree = sp.spatial.cKDTree(kernels.reshape((size, 5)),
                                        compact_nodes=True,
                                        balanced_tree=True)

            # Initialize threads, iterate every kernel
            threads = []
            for i in range(4):
                t = threading.Thread(name=str(i), target=self.kernelIteration,
                                     args=(i, hs, hr, k, kernels, newKernels,
                                           eps, kdTree, checks, 4, img, valid))
                t.setDaemon(True)  # Doesn't block main program
                threads.append(t)
                t.start()

            # Wait for threads to finish
            for t in threads:
                t.join()

            # Update kernels
            kernels = newKernels

            print "DEBUG: Checks min/avg/max:", checks.min(), \
                np.mean(checks), checks.max()

            # Maps check values to epsilon
            boolchecks = checks <= eps

            # Check how many kernels did not move
            stopped = np.count_nonzero(boolchecks)

            print "DEBUG: Fixed kernels:", stopped

            # Interrupt if limit is achieved
            if stopped >= (lim*size):
                break

        # Once done, copy the original pixel matrix
        kernelsCopy = kernels.astype("uint64")
        colors = img.pixels.copy()

        # Iterate every pixel, changing its color from the one given by its
        # kernel after iterating. Rounds down no matter the decimal part
        for j in np.arange(img.height):
            for i in np.arange(img.width):
                y, x = kernelsCopy[j][i][:2]
                img.pixels[j][i] = colors[y][x]

        # Store result in class variable
        img.updateImage()
        self.msFilter = img.copy()

    def meanshiftSegment(self, hr, filepath=None):

        '''This function should perform a segmentation in an image based on
        each pixel's color. It groups together neighbor pixels that have
        similar colors. By default, it uses the filtered image in the class
        instance, but it can load a file specified as a argument.'''

        # Check where to load image from, default is class instance
        if self.msFilter is None:
            # Check if filepath is specified
            if filepath is None:
                # Neither sources are valid, abort
                print "\nERROR: Please either " + \
                    "specify a file or run meanshiftFilter.\n"
                return
            else:
                # Load image from filepath
                img = PyImage()
                img.loadFile(filepath)
        else:
            # Load image from class
            img = self.msFilter.copy()

        # Start group matrix with zeros
        groups = np.zeros((img.height, img.width), dtype="int32")
        groups -= 1
        pixels = []  # List of pixels per group
        colors = []  # Average color per group
        lastGroup = 0

        # Iterate pixels, assigning group to each pixel
        for j in np.arange(img.height):
            for i in np.arange(img.width):

                # If pixel has no group, set a new group for it
                if groups[j][i] == -1:
                    groups[j][i] = lastGroup
                    lastGroup += 1
                    pixels.append([(j, i)])
                    colors.append(img.pixels[j][i].astype("float64"))

                # Get pixel neighbors
                neighbors = []
                if j:
                    neighbors.append((j-1, i, img.pixels[j-1][i]))
                    if i:
                        neighbors.append((j-1, i-1, img.pixels[j-1][i-1]))
                    if i < img.width - 1:
                        neighbors.append((j-1, i+1, img.pixels[j-1][i+1]))
                if j < img.height - 1:
                    neighbors.append((j+1, i, img.pixels[j+1][i]))
                    if i:
                        neighbors.append((j+1, i-1, img.pixels[j+1][i-1]))
                    if i < img.width - 1:
                        neighbors.append((j+1, i+1, img.pixels[j+1][i+1]))
                if i:
                    neighbors.append((j, i-1, img.pixels[j][i-1]))
                if i < img.width - 1:
                    neighbors.append((j, i+1, img.pixels[j][i+1]))

                # For each neighbor, check if color is similar
                group = groups[j][i]
                for neighbor in neighbors:
                    # Compute color difference
                    cDiff = colors[group] - neighbor[2]
                    cDiff = sum(cDiff ** 2)
                    cDiff **= 0.5
                    if cDiff < hr:
                        # Color is similar in all 3 channels, put neighbor as
                        # same group as current pixel
                        groups[neighbor[0]][neighbor[1]] = group
                        oldGroupLen = len(pixels[group])
                        pixels[group].append((neighbor[0], neighbor[1]))
                        color = colors[group]
                        color *= oldGroupLen
                        color += neighbor[2]
                        color /= oldGroupLen + 1
                        colors[group] = color

        print lastGroup

        # Iterate groups
        for g in range(lastGroup):
            # Iterate pixels, updating their colors
            color = colors[g].astype("uint8")
            for pixel in pixels[g]:
                img.pixels[pixel[0]][pixel[1]] = color

        # Store result
        self.msSegment = img
