from PIL import Image
from PIL import ImageDraw
from pyimage import PyImage
from gausspyramid import GaussPyramid

import threading
import numpy as np

# ------------------------------------------------------------------------
# Helper Matrix functions
# ------------------------------------------------------------------------


def invertMatrix(matrix):

    '''This function should return the inverse of the given matrix.'''

    # Compute determinant
    determinant = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # If det is zero, matrix is not invertible
    if not determinant:
        return None

    # Result is a 2x2 matrix from assumption of a 2x2 input
    result = np.empty((2, 2), dtype="float64")

    # Compute each individual cell
    result[0][0] = matrix[1][1] / determinant
    result[0][1] = -matrix[0][1] / determinant
    result[1][0] = -matrix[1][0] / determinant
    result[1][1] = matrix[0][0] / determinant

    return result


def minEigenValue(matrix):

    '''This function should return the smallest of a 2x2 matrix's
    eigenvalues.'''

    # Compute trace and determinant
    trace = matrix[0][0] + matrix[1][1]
    determinant = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # Even though a det of zero doesn't mean there are no eigenvalues, for the
    # purpose of this application we will discard any matrices that are not
    # invertible as well
    if determinant == 0:
        return None

    # Delta for l^2 - trace*l + det, where l is an eigenvalue
    delta = (trace ** 2) - (4 * determinant)

    # If delta is lesser than zero, there are no real eigenvalues
    if delta < 0:
        return None

    # Solve for l
    eigenA = (trace + (delta ** 0.5)) / 2
    eigenB = (trace - (delta ** 0.5)) / 2

    # Return lesser of computed eigenvalues
    return min(eigenA, eigenB)

# ------------------------------------------------------------------------
# Tracking point
# ------------------------------------------------------------------------


class TrackingPoint(object):

    '''This class is the structure that will hold together every information
    relating to a single point in the tracker.'''

    def __init__(self, j, i, matrixA, matrixB, eigen, color):

        self.prevPos = (0, 0)  # Previous (i, j) position
        self.deleteCounter = 0  # Counter for when point doesnt move much
        self.originalColor = int(color)  # Original color
        self.j = j  # Y position
        self.i = i  # X position
        self.matrixA = matrixA  # AtA matrix
        self.matrixB = matrixB  # AtB matrix
        self.eigen = eigen  # Lesser eigenvalue of A

    def updateMatrix(self, img, nextImg, scale=1, offset=[0, 0]):

        '''This function should update the point's matrices based on a new
        pair of images.'''

        # Offset for temporal derivative - map to int from float
        intOffset = map(lambda x: int(round(x)), offset)

        # Compute horizontal and vertical derivatives
        dx = img.simpleHorizontalDerivative()
        dy = img.simpleVerticalDerivative()

        # Apply scale to i and j, used by pyramid algorithm
        y = (self.j / scale)
        x = (self.i / scale)

        # Gauss filter for use later on. Assumed 3x3.
        gaussFilter = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype="float64")
        gaussFilter /= 16.0

        # Take windows of derivatives and compute temporal derivative
        dxWindow = dx[y-1:y+2, x-1:x+2]
        dyWindow = dy[y-1:y+2, x-1:x+2]
        dtWindow = img.temporalDerivative(nextImg, y, x, intOffset)

        if dxWindow.shape != (3, 3) or dyWindow.shape != (3, 3):
            self.matrixA = None
            return

        if dtWindow is None:
            self.matrixA = None
            return

        # Compute auxiliar values from windows, modified by gaussian filter
        dx2 = np.sum((dxWindow ** 2) * gaussFilter)
        dy2 = np.sum((dyWindow ** 2) * gaussFilter)
        dxdy = np.sum(dxWindow * dyWindow * gaussFilter)
        dxdt = np.sum(dxWindow * dtWindow * gaussFilter)
        dydt = np.sum(dyWindow * dtWindow * gaussFilter)

        # Set AtA matrix
        self.matrixA = np.array([[dx2, dxdy],
                                 [dxdy, dy2]])

        # Set AtB matrix
        self.matrixB = np.array([-dxdt, -dydt])

        # Compute AtA's lesser eigenvalue
        self.eigen = minEigenValue(self.matrixA)

    def runChecks(self, bounds, boxBounds, img):

        '''This function should check whether this point is still a valid point
        for tracking after its translation and matrix update.'''

        if self.matrixA is None:
            return False

        boundX = bounds[0]
        boundY = bounds[1]

        boxBoundX = boxBounds[0]
        boxBoundY = boxBounds[1]

        if self.i < boundX[0] or self.i > boundX[1]:
            return False

        if self.j < boundY[0] or self.j > boundY[1]:
            return False

        if self.i < boxBoundX[0] or self.i > boxBoundX[1]:
            return False

        if self.j < boxBoundY[0] or self.j > boxBoundY[1]:
            return False

        det = (self.matrixA[0][0]*self.matrixA[1][1] -
               self.matrixA[0][1]*self.matrixA[1][0])

        if det == 0:
            return False

        if abs(int(img.pixels[self.j][self.i]) - self.originalColor) > 25:
            return False

        ssd = ((self.i - self.prevPos[0])**2 + (self.j - self.prevPos[1])**2)
        if ssd < 1:
            self.deleteCounter += 1
            if self.deleteCounter > 4:
                return False

        return True

    def computeFlux(self):

        '''This function should compute the flux given by the matrices
        currently stored.'''

        # Invert AtA matrix
        invA = invertMatrix(self.matrixA)

        if invA is None:
            return None, None

        # Solve flux = ((AtA)^-1)AtB
        fluxX = np.sum(invA[0] * self.matrixB)
        fluxY = np.sum(invA[1] * self.matrixB)

        # Update flux
        self.fluxY = fluxY
        self.fluxX = fluxX

        return fluxY, fluxX

    def translate(self):

        '''This function should translate the point by the flux to be computed
        by the matrices stored.'''

        # Translates point to new pixel in image
        self.prevPos = (self.i, self.j)
        self.j += int(round(self.fluxY))
        self.i += int(round(self.fluxX))

        # Reset flux
        self.fluxY = 0
        self.fluxX = 0

# ------------------------------------------------------------------------
# Tracker
# ------------------------------------------------------------------------


class KLTTracker(object):

    '''This class is the structure that will hold together all of the tracking
    information throughout its execution, as well as harbour all of the needed
    functions to perform the tracking.'''

    def __init__(self):

        self.nFrames = 0  # Number of frames to be processed
        self.frames = []  # Original frames
        self.grayscaleFrames = []  # Grayscale frames
        self.corners = []  # Selected tracking regions
        self.points = []  # Selected tracking points
        self.pyramid = 0  # Number of pyramid levels to go down
        self.pyramids = []  # Frame pyramids
        self.filepath = None  # Filepath to save to

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFile(self, filepath, nFrames, pyramid=0):

        '''This function should load frames of a video with a regular file
        expression to read filenames. The number of frames is known beforehand
        to ease computing.'''

        # Update variables based on input
        self.pyramid = pyramid
        self.nFrames = nFrames
        self.filepath = filepath

        # Separate name from extension
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        # For each frame that will be processed, we will load a single file
        for i in np.arange(1, nFrames+1):

            print "DEBUG: Loading frame", i

            # Load each frame
            frame = PyImage()
            frame.loadFile(path+str(i)+extension)
            frame.img = frame.img.convert("RGB")
            frame.updatePixels()

            # Compute its grayscale counterpart
            grayscaleFrame = frame.copy()
            grayscaleFrame.img = grayscaleFrame.img.convert("L")
            grayscaleFrame.updatePixels()

            # Append frames to lists
            self.frames.append(frame)
            self.grayscaleFrames.append(grayscaleFrame)

            # If pyramids will be used, start a pyramid for every grayscale
            # frame to be used later on
            if pyramid:
                print "DEBUG: Computing pyramids for frame", i
                framePyramid = GaussPyramid(pyramid)
                framePyramid.loadImage(grayscaleFrame.img)
                framePyramid.reduceMax()
                self.pyramids.append(framePyramid)

    def saveFrame(self, index):

        '''This function should save a single processed frame back into the
        directory it came from. In the processed frame should be pictured the
        tracking points used for tracking, the computed flow for each point for
        that frame, and the highlighted tracking area.'''

        # Separate name from extension
        path = self.filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-out-'
        filepath = path+str(index)+extension

        # Get frame's pixels
        frame_pixels = self.frames[index].pixels

        # Identify points in blue
        for point in self.points:
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    if not y and not x:
                        continue
                    frame_pixels[point.j+y][point.i+x] = np.array([0, 0, 255])

        # Build image from modified pixels, draw lines on it based on flow
        im = Image.fromarray(frame_pixels, "RGB")
        draw = ImageDraw.Draw(im)
        for point in self.points:
            start = (point.i, point.j)
            end = (point.i + int(round(point.fluxX)),
                   point.j + int(round(point.fluxY)))
            draw.line([start, end], fill=(0, 255, 0))

        # Get corners for region marking and draw lines
        tlCorner = self.corners[index][0]
        brCorner = self.corners[index][1]
        trCorner = (brCorner[0], tlCorner[1])
        blCorner = (tlCorner[0], brCorner[1])
        draw.line([tlCorner, trCorner], fill=(255, 0, 0))
        draw.line([tlCorner, blCorner], fill=(255, 0, 0))
        draw.line([trCorner, brCorner], fill=(255, 0, 0))
        draw.line([blCorner, brCorner], fill=(255, 0, 0))

        # Save final image
        im.save(filepath)

    # ------------------------------------------------------------------------
    # Tracking functions
    # ------------------------------------------------------------------------

    def filterPoints(self):

        '''This function should receive a list of tracking points and filter
        them so that no points are within a 1 pixel radius from the other.'''

        # Sort points by eigenvalue
        self.points.sort(key=lambda x: x.eigen, reverse=True)

        # Start a set of points to delete
        delete = set()
        nPoints = len(self.points)

        # Iterate every point, checking if it is a local maximum for eignevalue
        for i in range(nPoints):
            # Check if point is already marked for deletion
            if i in delete:
                continue
            # Iterate every point with a lower eigenvalue
            for j in range(i+1, nPoints):
                # Compute horizontal and vertical distance to check if points
                # are neighbors
                hDiff = abs(self.points[i].i - self.points[j].i)
                vDiff = abs(self.points[i].j - self.points[j].j)
                if hDiff < 2 or vDiff < 2:
                    # Points are neighbors, delete the second one
                    delete.add(j)

        # Effectively delkete excess points
        for index in sorted(delete, reverse=True):
            del(self.points[index])

    def findFeaturePoints(self, img, nextImg, corners):

        '''This function should find the best feature points to be used for
        tracking the selected area throughout the frames.'''

        print "DEBUG: Finding feature points"

        # Compute horizontal and vertical derivatives
        dx = img.simpleHorizontalDerivative()
        dy = img.simpleVerticalDerivative()

        # Take in a corner to limit pixels
        start = corners[0]
        end = corners[1]

        # Gaussian filter for later, assumed to be 3x3
        gaussFilter = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype="float64")
        gaussFilter /= 16.0

        # List of eigenvalues for sorting points
        eigenValues = []

        # Iterate every pixel, checking if it is a good candidate for tracking
        for j in np.arange(start[1], end[1]+1):
            for i in np.arange(start[0], end[0]+1):

                # If pixel will become a border pixel when applying pyramid
                # reduction, discard it
                if (j < (2**self.pyramid)+1 or
                        i < (2**self.pyramid)+1 or
                        j > (img.height - 2 - (2**self.pyramid)) or
                        i > (img.width - 2 - (2**self.pyramid))):
                    continue

                # Compute window derivatives and temporal dereivative
                dxWindow = dx[j-1:j+2, i-1:i+2]
                dyWindow = dy[j-1:j+2, i-1:i+2]
                dtWindow = img.temporalDerivative(nextImg, j, i, [0, 0])

                # Compute helper values with gaussian filter applied
                dx2 = np.sum((dxWindow ** 2) * gaussFilter)
                dy2 = np.sum((dyWindow ** 2) * gaussFilter)
                dxdy = np.sum(dxWindow * dyWindow * gaussFilter)
                dxdt = np.sum(dxWindow * dtWindow * gaussFilter)
                dydt = np.sum(dyWindow * dtWindow * gaussFilter)

                # Compute AtA matrix
                matrixA = np.array([[dx2, dxdy],
                                    [dxdy, dy2]])

                # Compute AtB matrix
                matrixB = np.array([-dxdt, -dydt])

                # Compute smaller eigenvalue for AtA
                minEigen = minEigenValue(matrixA)
                if minEigen is not None:
                    # If there are eigenvalues, add pixel as a tracking point
                    eigenValues.append(minEigen)
                    self.points.append(TrackingPoint(j, i,
                                                     matrixA,
                                                     matrixB,
                                                     minEigen,
                                                     img.pixels[j][i]))

        # Compute highest eigenvalue
        maxEigen = max(eigenValues)
        eigenThresh = 0.1 * maxEigen

        # Discard points whose eigenvalue is too small
        for point in self.points:
            if point.eigen < eigenThresh:
                self.points.remove(point)

        print "DEBUG: Filtering feature points"

        # Further filter points
        self.filterPoints()

    def computeAverageFlux(self, fluxSumY, fluxSumX, nPoints):

        '''This function computes the average flux based on the sum of every
        point's flux.'''

        # Compute average flux
        avgFluxY = fluxSumY / nPoints
        avgFluxX = fluxSumX / nPoints

        # Update border of tracking region
        border = self.corners[-1]
        tlY = border[0][1] + int(round(avgFluxY))
        brY = border[1][1] + int(round(avgFluxY))
        tlX = border[0][0] + int(round(avgFluxX))
        brX = border[1][0] + int(round(avgFluxX))
        self.corners.append(((tlX, tlY), (brX, brY)))

    def movePoints(self, frameIndex):

        '''This function should move the feature points to the next frame in
        the list, updating the selected area as well.'''

        # Start flux sum variables and compute number of points
        fluxSumY = fluxSumX = 0
        nPoints = len(self.points)
        width = self.frames[frameIndex].width
        height = self.frames[frameIndex].height
        p = self.pyramid**2

        bounds = ((p+1, width-p-1), (p+1, height-p-1))
        corner = self.corners[-1]
        boxBounds = ((corner[0][0]-20, corner[1][0]+20),
                     (corner[0][1]-20, corner[1][1]+20))

        if self.pyramid:
            # If pyramids will be used, use the pyramid algorithm to update
            # each point's matrix and then move points
            for point in self.points:
                fluxY = fluxX = 0
                for level in reversed(range(len(
                        self.pyramids[frameIndex].pyramid))):
                    # For each level, we start by doubling the previous level's
                    # flux value, since it was a halved scale
                    fluxY *= 2
                    fluxX *= 2
                    # Update matrix with previous flux as offset
                    point.updateMatrix(self.pyramids[frameIndex]
                                           .pyramid[level],
                                       self.pyramids[frameIndex+1]
                                           .pyramid[level],
                                       2**level,
                                       [fluxY, fluxX])
                    if point.matrixA is None:
                        break
                    # Compute flux
                    y, x = point.computeFlux()
                    if y is None:
                        point.matrixA = None
                        break
                    fluxY += y
                    fluxX += x
                point.fluxX = fluxX
                point.fluxY = fluxY
                # Add flux to flux sum
                checkOk = point.runChecks(bounds, boxBounds,
                                          self.grayscaleFrames[frameIndex])
                if not checkOk:
                    self.points.remove(point)
                    continue
                fluxSumY += fluxY
                fluxSumX += fluxX
            # Compute average flux and save the current frame
            self.computeAverageFlux(fluxSumY, fluxSumX, len(self.points))
            self.saveFrame(frameIndex)
            # Translate every point
            for point in self.points:
                point.translate()

        else:
            # Else, simply move points and then update matrix to the next pair
            # of frames
            for point in self.points:
                # Compute flux and add it to flux total sum
                fluxY, fluxX = point.computeFlux()
                checkOk = point.runChecks(bounds, boxBounds,
                                          self.grayscaleFrames[frameIndex])
                if not checkOk:
                    self.points.remove(point)
                    continue
                fluxSumY += fluxY
                fluxSumX += fluxX
            # Compute average flux and save the current frame
            self.computeAverageFlux(fluxSumY, fluxSumX, len(self.points))
            self.saveFrame(frameIndex)
            # Update matrix with the next pair of frames, and translate point
            for point in self.points:
                point.translate()
                if frameIndex != self.nFrames - 2:
                    point.updateMatrix(self.grayscaleFrames[frameIndex+1],
                                       self.grayscaleFrames[frameIndex+2])
                    if point.matrixA is None:
                        self.points.remove(point)
                        continue

    def trackRegion(self, topLeftCorner, bottomRightCorner):

        '''This function should permorm the tracking of a specific region
        along the series of frames loaded.'''

        borders = [topLeftCorner, bottomRightCorner]
        self.corners.append(borders)

        self.findFeaturePoints(self.grayscaleFrames[0],
                               self.grayscaleFrames[1],
                               self.corners[-1])

        nPoints = len(self.points)

        for i in range(self.nFrames-1):
            print "DEBUG: Processing frame", i
            self.movePoints(i)
            if len(self.points) < 0.75*nPoints:
                print "DEBUG: Lost too many points, computing them again"
                self.points = []
                try:
                    self.findFeaturePoints(self.grayscaleFrames[i+1],
                                           self.grayscaleFrames[i+2],
                                           self.corners[-1])
                    nPoints = len(self.points)
                except IndexError:
                    pass

        self.corners.append(self.corners[-1])
        print "Save"
        self.saveFrame(self.nFrames-1)
