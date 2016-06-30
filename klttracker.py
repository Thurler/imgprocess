from PIL import Image
from pyimage import PyImage
from gausspyramid import GaussPyramid

import threading
import numpy as np

# ------------------------------------------------------------------------
# Helper Matrix functions
# ------------------------------------------------------------------------


def invertMatrix(self, matrix):

    '''This function should return the inverse of the given matrix.'''

    determinant = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    if not determinant:
        return None

    result = np.empty((2, 2), dtype="float64")

    result[0][0] = matrix[1][1] / determinant
    result[0][1] = -matrix[0][1] / determinant
    result[1][0] = -matrix[1][0] / determinant
    result[1][1] = matrix[0][0] / determinant

    return result


def minEigenValue(self, matrix):

    '''This function should return the smallest of a 2x2 matrix's
    eigenvalues.'''

    trace = matrix[0][0] + matrix[1][1]
    determinant = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    delta = (trace ** 2) - (4 * determinant)

    if delta < 0:
        return None

    eigenA = (trace + (delta ** 0.5)) / 2
    eigenB = (trace - (delta ** 0.5)) / 2

    return min(eigenA, eigenB)

# ------------------------------------------------------------------------
# Tracking point
# ------------------------------------------------------------------------


class TrackingPoint(object):

    '''This class is...'''

    def __init__(self, j, i, matrixA, matrixB, eigen):

        self.j = j
        self.i = i
        self.matrixA = matrixA
        self.matrixB = matrixB
        self.eigen = eigen

    def updateMatrix(self, img, nextImg, scale=1, offset=[0, 0]):

        '''This function should update the point's matrices based on a new
        pair of images.'''

        dx = img.simpleHorizontalDerivative()
        dy = img.simpleVerticalDerivative()
        dt = img.temporalDerivative(nextImg)

        y = (self.j / scale)
        x = (self.i / scale)

        gaussFilter = np.array([1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1], dtype="float64")
        gaussFilter /= 16.0

        dxWindow = dx[y-1:y+2, x-1:x+2]
        dyWindow = dy[y-1:y+2, x-1:x+2]

        y += offset[0]
        x += offset[1]
        dtWindow = dt[y-1:y+2, x-1:x+2]

        dx2 = sum((dxWindow ** 2) * gaussFilter)
        dy2 = sum((dyWindow ** 2) * gaussFilter)
        dxdy = sum(dxWindow * dyWindow * gaussFilter)
        dxdt = sum(dxWindow * dtWindow * gaussFilter)
        dydt = sum(dyWindow * dtWindow * gaussFilter)

        self.matrixA = np.array([dx2, dxdy],
                                [dxdy, dy2])

        self.matrixB = np.array([-dxdt, -dydt])

        self.eigen = minEigenValue(matrixA)

    def computeFlux(self):

        '''This function should compute the flux given by the matrices
        currently stored.'''

        invA = invertMatrix(matrixA)
        fluxX = sum(invA[0] * matrixB)
        fluxY = sum(invA[1] * matrixB)

        return fluxY, fluxX

    def translate(self):

        '''This function should translate the point by the flux to be computed
        by the matrices stored.'''

        fluxY, fluxX = self.computeFlux()
        self.j += fluxY
        self.i += fluxX

        return fluxY, fluxX

# ------------------------------------------------------------------------
# Tracker
# ------------------------------------------------------------------------


class KLTTracker(object):

    '''This class is...'''

    def __init__(self):

        self.frames = []
        self.grayscaleFrames = []
        self.corners = []
        self.points = []
        self.pyramid = None
        self.pyramids = []

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFile(self, filepath, nFrames, pyramid=False):

        '''This function should load frames of a video with a regular file
        expression to read filenames. All frames are assumed to be in the
        "frames" directory in the cwd. The number of frames is known beforehand
        to ease computing.'''

        self.frames = []
        self.pyramid = pyramid

        # Separate name from extension
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        for i in np.arange(nFrames):
            frame = PyImage()
            frame.loadFile(path+str(i)+extension)
            grayscaleFrame = frame.copy()
            grayscaleFrame.img.convert("L")
            grayscaleFrame.updatePixels()
            self.frames.append(frame)
            self.grayscaleFrames.append(grayscaleFrame)

            if pyramid:
                framePyramid = GaussPyramid(pyramid)
                framePyramid.loadImage(grayscaleFrame)
                framePyramid.reduceMax()
                self.pyramids.append(framePyramid)

    def saveFile(self, filepath):

        '''This function should save the frames accepted as input marked with
        the tracking points used and the tracking area highlighted.'''

        # Separate name from extension
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        # TODO

    # ------------------------------------------------------------------------
    # Tracking functions
    # ------------------------------------------------------------------------

    def filterPoints(self):

        '''This function should receive a list of tracking points and filter
        them so that no points are within a 1 pixel radius from the other.'''

        self.points.sort(key=lambda x: x.eigen, reverse=True)

        delete = []
        nPoints = len(self.points)
        for i in range(nPoints):
            if i in delete:
                continue
            for j in range(i+1, nPoints):
                hDiff = abs(self.points[i].i - self.points[j].i)
                vDiff = abs(self.points[i].j - self.points[j].j)
                if hDiff < 2 or vDiff < 2:
                    delete.append(j)

        for index in delete:
            del(self.points[index])

    def findFeaturePoints(self, img, nextImg, corners):

        '''This function should find the best feature points to be used for
        tracking the selected area throughout the frames.'''

        dx = img.simpleHorizontalDerivative()
        dy = img.simpleVerticalDerivative()
        dt = img.temporalDerivative(nextImg)

        start = corners[0]
        end = corners[1]

        gaussFilter = np.array([1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1], dtype="float64")
        gaussFilter /= 16.0

        eigenValues = []

        for j in np.arange(start[0], end[0]+1):
            for i in np.arange(start[1], end[1]+1):

                dxWindow = dx[j-1:j+2, i-1:i+2]
                dyWindow = dy[j-1:j+2, i-1:i+2]
                dtWindow = dt[j-1:j+2, i-1:i+2]

                dx2 = sum((dxWindow ** 2) * gaussFilter)
                dy2 = sum((dyWindow ** 2) * gaussFilter)
                dxdy = sum(dxWindow * dyWindow * gaussFilter)
                dxdt = sum(dxWindow * dtWindow * gaussFilter)
                dydt = sum(dyWindow * dtWindow * gaussFilter)

                matrixA = np.array([dx2, dxdy],
                                   [dxdy, dy2])

                matrixB = np.array([-dxdt, -dydt])

                minEigen = minEigenValue(matrixA)
                if minEigen is not None:
                    eigenValues.append(minEigen)
                    self.points.append(TrackingPoint(j, i,
                                                     matrixA,
                                                     matrixB,
                                                     eigen))

        maxEigen = max(eigenValues)
        eigenThresh = 0.1 * maxEigen

        for point in self.points:
            if point.eigen < eigenThresh:
                self.points.remove(point)

        self.filterPoints()

    def movePoints(self, frameIndex):

        '''This function should move the feature points to the next frame in
        the list, updating the selected area as well. It behaves differently
        in the presence of pyramids: if there are none, it computes the flux
        and updates matrices; else, it updates matrices and then computes the
        flux. This is because for no pyramids we can use the matrice from the
        feature points selection.'''

        fluxSumY = fluxSumX = 0
        nPoints = len(self.points)

        if self.pyramid:
            for point in self.points:
                fluxY = fluxX = 0
                for level in reversed(range(len(
                        self.pyramids[frameIndex].pyramid))):
                    fluxY *= 2
                    fluxX *= 2
                    point.updateMatrix(self.pyramids[frameIndex]
                                           .pyramid[level],
                                       self.pyramids[frameIndex+1]
                                           .pyramid[level],
                                       2**level,
                                       [fluxY, fluxX])
                    y, x = point.translate()
                    fluxY += y
                    fluxX += x
                fluxSumY += fluxY
                fluxSumX += fluxX

        else:
            for point in self.points:
                fluxY, fluxX = point.translate()
                fluxSumY += fluxY
                fluxSumX += fluxX
                if frameIndex != self.nFrames - 1:
                    point.updateMatrix(self.grayscaleFrames[frameIndex+1],
                                       self.grayscaleFrames[frameIndex+2])

        avgFluxY = fluxSumY / nPoints
        avgFluxX = fluxSumX / nPoints

        border = self.corners[-1]
        border[0][0] += avgFluxY
        border[1][0] += avgFluxY
        border[0][1] += avgFluxX
        border[1][1] += avgFluxX

        self.corners.append(border)

    def trackRegion(self, topLeftCorner, bottomRightCorner):

        '''This function should permorm the tracking of a specific region
        along the series of frames loaded.'''

        borders = [topLeftCorner, bottomRightCorner]
        self.corners.append(borders)

        findFeaturePoints(self.grayscaleFrames[0],
                          self.grayscaleFrames[1],
                          corners)
