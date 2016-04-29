from PIL import Image
from pyimage import PyImage
from gausspyramid import GaussPyramid
from laplacepyramid import LaplacePyramid

import numpy as np


class LaplaceBlender(object):

    '''This class represents the Laplacian Pyramid Blending process in a data
    structure containing two laplacian pyramids, for two given images, each of
    those with a gaussian pyramid as well, a third gaussian pyramid for a given
    mask that will be used to blend, and a third laplacian pyramid for the
    blending of the other two laplacian pyramids, given the mask.'''

    def __init__(self, reduce_default=4):

        self.l_pyramid_a = LaplacePyramid(reduce_default)
        self.l_pyramid_b = LaplacePyramid(reduce_default)
        self.g_pyramid_mask = GaussPyramid(reduce_default)

        self.l_pyramid_blend = None

    # ------------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------------

    def defaultMask(self, size, mode):

        '''This function should generate a default mask image and return it.
        The image generated has size and mode specified as arguments. The
        default mask is a binary mask that is white in the left half and black
        in the right half.'''

        # Invert x,y for numpy
        size = (size[1], size[0])
        # Set matrix of zeros
        mask = np.zeros(size)
        # Fill rightmost half with ones
        mask[:, :size[1]/2] = np.ones((size[0],
                                       size[1]/2))
        # Multiply everything by 255
        mask *= 255

        # Convert image to grayscale mode, then the desired mode, and return
        return Image.fromarray(mask.astype("uint8"), 'L').convert(mode)

    # ------------------------------------------------------------------------
    # Input and Output functions
    # ------------------------------------------------------------------------

    def loadFiles(self, filepath_a, filepath_b, filepath_mask=None):

        '''This function should load at least two files that contain images to
        be used in the blending. A third file can be passed in to be used as
        mask. This function should initiallize laplacian pyramids for both
        mandatory images, and a gaussian pyramid for the mask. If a mask is not
        given, a default one is generated.'''

        # Load files into laplacian pyramids
        self.l_pyramid_a.loadFile(filepath_a)
        self.l_pyramid_b.loadFile(filepath_b)

        # Get size and mode for both of them
        size_a = self.l_pyramid_a.gauss_pyramid.pyramid[0].img.size
        size_b = self.l_pyramid_b.gauss_pyramid.pyramid[0].img.size
        mode_a = self.l_pyramid_a.gauss_pyramid.pyramid[0].img.mode
        mode_b = self.l_pyramid_b.gauss_pyramid.pyramid[0].img.mode

        # Assert images have same size and pixel mode
        if size_a != size_b:
            print "\nERROR: The two images do not have the same size\n"
            return

        if mode_a != mode_b:
            print "\nERROR: The images do not have same number of channels\n"
            return

        # Build the laplacian pyramid for both images
        self.l_pyramid_a.buildPyramid()
        self.l_pyramid_b.buildPyramid()

        if filepath_mask is None:
            # Generate default mask and generate gaussian pyramid for it
            img = self.defaultMask(size_a, mode_a)
            self.g_pyramid_mask.loadImage(img)
            self.g_pyramid_mask.reduceMax()

        else:
            # Read mask from disk and generate gaussian pyramid for it
            self.g_pyramid_mask.loadFile(filepath_mask)
            self.g_pyramid_mask.pyramid[-1].img = \
                self.g_pyramid_mask.pyramid[-1].img.convert(mode_a)
            self.g_pyramid_mask.pyramid[-1].updatePixels()
            self.g_pyramid_mask.reduceMax()

    def loadImages(self, image_a, image_b, image_mask=None):

        '''This function should load at least two images that contain images to
        be used in the blending. A third image can be passed in to be used as
        mask. This function should initiallize laplacian pyramids for both
        mandatory images, and a gaussian pyramid for the mask. If a mask is not
        given, a default one is generated.'''

        # Load files into laplacian pyramids
        self.l_pyramid_a.loadImage(image_a)
        self.l_pyramid_b.loadImage(image_b)

        # Get size and mode for both of them
        size_a = self.l_pyramid_a.gauss_pyramid.pyramid[0].img.size
        size_b = self.l_pyramid_b.gauss_pyramid.pyramid[0].img.size
        mode_a = self.l_pyramid_a.gauss_pyramid.pyramid[0].img.mode
        mode_b = self.l_pyramid_b.gauss_pyramid.pyramid[0].img.mode

        # Assert images have same size and pixel mode
        if size_a != size_b:
            print "\nERROR: The two images do not have the same size\n"
            return

        if mode_a != mode_b:
            print "\nERROR: The images do not have same number of channels\n"
            return

        # Build the laplacian pyramid for both images
        self.l_pyramid_a.buildPyramid()
        self.l_pyramid_b.buildPyramid()

        if filepath_mask is None:
            # Generate default mask and generate gaussian pyramid for it
            img = self.defaultMask(size_a, mode_a)
            self.g_pyramid_mask.loadImage(img)
            self.g_pyramid_mask.reduceMax()

        else:
            # Read mask and generate gaussian pyramid for it
            image_mask.convert(mode_a)
            self.g_pyramid_mask.loadImage(image_mask)
            self.g_pyramid_mask.reduceMax()

    def saveFile(self, filepath):

        '''This function should save every pyramid it has into files.'''

        # Separate filename from extension
        path = filepath.split('.')
        extension = '.' + path[-1]
        path = "".join(path[:-1]) + '-'

        # Save all pyramids
        self.l_pyramid_a.savePyramid(path+'lpyramid-A'+extension)
        self.l_pyramid_b.savePyramid(path+'lpyramid-B'+extension)
        self.g_pyramid_mask.savePyramid(path+'gpyramid-Mask'+extension)

        if self.l_pyramid_blend is not None:
            self.l_pyramid_blend.savePyramid(path+'lpyramid-blend'+extension)

    # ------------------------------------------------------------------------
    # Blending functions
    # ------------------------------------------------------------------------

    def blendPyramids(self):

        '''This function should blend pyramids A and B into pyramid BLEND.'''

        # Assert pyramids have been loaded
        if not self.g_pyramid_mask.pyramid:
            print "\nERROR: Please load images onto the blender\n"

        # Get image to start new pyramid's gaussian pyramid from pyramid A
        img = self.l_pyramid_a.gauss_pyramid.pyramid[-1]

        # Blend it with the one from pyramid B using the corresponding mask
        img = img.blend(self.l_pyramid_b.gauss_pyramid.pyramid[-1],
                        self.g_pyramid_mask.pyramid[-1])

        # Generate new laplace pyramid blending A and B
        self.l_pyramid_blend = self.l_pyramid_a.blend(self.l_pyramid_b,
                                                      self.g_pyramid_mask,
                                                      img)

    def collapse(self, filepath):

        '''This function should collapse the blended pyramid.'''

        # Assert blended image is laoded
        if not self.l_pyramid_blend.pyramid:
            print "\nERROR: Please blend the image pyramids first\n"

        # Collapse it
        self.l_pyramid_blend.collapsePyramid(filepath,
                                             self.g_pyramid_mask.info_loss)
