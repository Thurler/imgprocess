from PIL import Image
from pyimage import PyImage
from gausspyramid import GaussPyramid
from laplacepyramid import LaplacePyramid

import numpy as np


class LaplaceBlender(object):

    '''This class...'''

    def __init__(self):

        self.l_pyramid_a = LaplacePyramid()
        self.l_pyramid_b = LaplacePyramid()
        self.g_pyramid_mask = GaussPyramid()

        self.l_pyramid_blend = LaplacePyramid()

    # ------------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------------

    def defaultMask(self):

        '''This function should...'''

        mask = np.zeros(size_a)
        mask[:, :size_a[1]/2] = np.ones((size_a[0],
                                         size_a[1]/2))
        mask *= 255
        return Image.fromarray(mask.astype("uint8"), "L")

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFiles(self, filepath_a, filepath_b, filepath_mask=None):

        '''This function should...'''

        self.l_pyramid_a.loadFile(filepath_a)
        self.l_pyramid_b.loadFile(filepath_b)

        size_a = self.l_pyramid_a.gauss_pyramid.pyramid[0].size
        size_b = self.l_pyramid_b.gauss_pyramid.pyramid[0].size

        if size_a != size_b:
            print "\nERROR: The two images do not have the same size\n"

        if filepath_mask is None:
            img = defaultMask()
            self.g_pyramid_mask.loadImage(img)
            self.g_pyramid_mask.reduceMax()

        else:
            self.g_pyramid_mask.loadFile(filepath_mask)
            self.g_pyramid_mask.reduceMax()

    # def loadImages(self, image_a, image_b, image_mask=None)
