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

        self.l_pyramid_blend = None

    # ------------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------------

    def defaultMask(self, size):

        '''This function should...'''

        mask = np.zeros(size)
        mask[:, :size[1]/2] = np.ones((size[0],
                                       size[1]/2))
        mask *= 255

        return Image.fromarray(mask.astype("uint8"), "L")

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFiles(self, filepath_a, filepath_b, filepath_mask=None):

        '''This function should...'''

        self.l_pyramid_a.loadFile(filepath_a)
        self.l_pyramid_b.loadFile(filepath_b)

        size_a = self.l_pyramid_a.gauss_pyramid.pyramid[0].img.size
        size_b = self.l_pyramid_b.gauss_pyramid.pyramid[0].img.size

        if size_a != size_b:
            print "\nERROR: The two images do not have the same size\n"
            return

        if filepath_mask is None:
            img = self.defaultMask(size_a)
            self.g_pyramid_mask.loadImage(img)
            self.g_pyramid_mask.reduceMax()

        else:
            self.g_pyramid_mask.loadFile(filepath_mask)
            self.g_pyramid_mask.pyramid[-1].img.convert('L')
            self.g_pyramid_mask.pyramid[-1].updatePixels()
            self.g_pyramid_mask.reduceMax()

    def loadImages(self, image_a, image_b, image_mask=None):

        '''This function should...'''

        self.l_pyramid_a.loadImage(image_a)
        self.l_pyramid_b.loadImage(image_b)

        size_a = self.l_pyramid_a.gauss_pyramid.pyramid[0].img.size
        size_b = self.l_pyramid_b.gauss_pyramid.pyramid[0].img.size

        if size_a != size_b:
            print "\nERROR: The two images do not have the same size\n"
            return

        if filepath_mask is None:
            img = self.defaultMask(size_a)
            self.g_pyramid_mask.loadImage(img)
            self.g_pyramid_mask.reduceMax()

        else:
            image_mask.convert('L')
            self.g_pyramid_mask.loadImage(image_mask)
            self.g_pyramid_mask.reduceMax()

    def saveFile(self, filepath):

        '''This function should...'''

        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        self.l_pyramid_a.saveFile(path+'lpyramid-A'+extension)
        self.l_pyramid_b.saveFile(path+'lpyramid-B'+extension)
        self.g_pyramid_mask.saveFile(path+'gpyramid-Mask'+extension)

        if self.l_pyramid_blend is not None:
            self.l_pyramid_blend.saveFile(path+'lpyramid-blend'+extension)

    # ------------------------------------------------------------------------
    # Blending functions
    # ------------------------------------------------------------------------

    def blendPyramids(self):

        '''This function should...'''

        if not self.g_pyramid_mask.pyramid:
            print "\nERROR: Please load images onto the blender\n"

        img = self.l_pyramid_a.gauss_pyramid.pyramid[-1]

        img = img.blend(self.l_pyramid_b.gauss_pyramid.pyramid[-1],
                        self.g_pyramid_mask.pyramid[-1])

        self.l_pyramid_blend = self.l_pyramid_a.blend(self.l_pyramid_b,
                                                      self.g_pyramid_mask)
