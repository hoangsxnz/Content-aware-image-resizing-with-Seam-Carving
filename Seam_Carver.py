from logging import raiseExceptions, warn
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import cv2
import numba
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')

class Seam_Carver:
    '''
    Variable:
        _origin_image: raw image before any use of crop or enlarge
        _image: image after use of any crop or enlarge

    Method:
        energy_image:       Compute gradient(energy) of image by sober filter
                                ==> Return: energy image
        seam_matrix_calculate:     compute minimum seam line by dynamic programming(dp)
                                ==> Return: track matrix(dp to track seam line)
                                            backtrack matrix(to backtracking matrix track ti find minimum)
        minimum_seam:       compute minimum ver l seam line
                                ==> Return: matrix mask(True if pixel lie on seam line)
        seam_column_delete: delete one vertical seam line
                                ==> Return image after delete seam line
        crop_column:        delete multiple vertical seam line 
                                ==> Return image after crop
        crop_row:           delete multiple horizontal seam line
                                ==> Return image after crop
        restore_image:      set seam carving image to original image
                                ==> Return original image
        crop:               crop image along both size
                                ==> Return cropped image
        crop_square:        crop image to square image
                                ==> Return cropped image
    '''


    ################################
    #  SOBER FILTER
    ################################

    #filter for x axis
    __filter_x = np.array([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
    ])
    # Stack 3 filter for 3 channel Red, Green, Blue
    __filter_x = np.stack([__filter_x] * 3, axis=2)

    #filter for y axis
    __filter_y = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # Stack 3 filter for 3 channel Red, Green, Blue
    __filter_y = np.stack([__filter_y] * 3, axis=2)


    def __init__(self, image = None):
        self._origin_image = image
        self._image = image.copy()


    ################################
    # ENERGY CALCULATION
    ################################

    def energy_image(self, visual = 0):
        ''' Calculate and return energy of this object image by sober filter. 

        Parameter:
            visual: 0 for default, change to 1 to visualize energy image before return
        
        Return:
            energy matrix (same size with image)
        '''
        assert visual in [0,1], 'visual must be 0 or 1, found {}'.format(visual)
        # print(self._image)
        image = np.array(self._image, dtype = 'float')
        dx = convolve(image, self.__filter_x)
        dy = convolve(image, self.__filter_y)

        conv = np.sqrt(dx**2 + dy**2)
        energy = np.sum(conv, axis = 2)
        # print(energy)

        if visual != 0:
            a, b = energy.shape
            plt.figure(figsize = (min(10*b/a, 20), 10))
            plt.imshow(energy, cmap = 'gray')
            plt.title(str(energy.shape))
            plt.show()
        return energy


    ################################
    # SEAM MATRIX CALCULATION
    ################################

    @numba.jit
    def seam_matrix_calculate(self):
        ''' Calculate seam matrix.
            
        Return:
            track: seam matrix
            backtrack: use to backtracking seam line
        '''
        nrows, ncols, _ = self._image.shape
        image_energy = self.energy_image()

        track = np.copy(image_energy)
        backtrack = np.zeros_like(track, dtype = np.int)

        for i in range(1, nrows):
            for j in range(ncols):
                if j == 0:
                    idx = np.argmin(track[i-1, j:j+2])
                    backtrack[i, j] = idx + j
                    track[i, j] += track[i-1, j + idx]
                else:
                    idx = np.argmin(track[i-1, j-1:j+2])
                    backtrack[i, j] = idx + j-1
                    track[i, j] += track[i-1, idx + j-1]
        return track, backtrack
    

    ################################
    # FINDING VERTICAL SEAM LINE
    ################################

    @numba.jit
    def minimum_seam(self, row = 0):
        ''' Using seam_calculate() method to compute minimum vertical seam line.
        
        Parameter:
            row: 0 for default. Change to 1 to compute horizontal seam instead of vertical
        
        Return: 
            mask: bool matrix same size with image. True if pixel is lie on seam line
        '''
        assert row in [0,1], 'row must be 0 or 1, found {}'.format(row)

        if row:
            self._image = np.rot90(self._image, 1)

        track, backtrack = self.seam_matrix_calculate()        
        nrows, ncols, _ = self._image.shape
        mask = np.zeros((nrows, ncols), dtype = np.bool)

        j = np.argmin(track[-1])
        for i in range(nrows-1, -1, -1):
            mask[i, j] = True 
            j = backtrack[i, j]

        if row:
            mask = np.rot90(mask, 3)
            self._image = np.rot90(self._image, 3)

        mask = np.stack([mask]*3, axis = 2)
        return mask 


    ################################
    # DELETE ONE VERTICAL SEAM
    ################################

    @numba.jit
    def seam_column_delete(self, visual = 0, row = 0):
        ''' Delete minimum vertical seam line.
        
        Parameter:
            visual: 0 for default. Change to 1 to just visual minimum seam line and don't delete
        
        Return:
            image with 1 less column than original image
        '''
        assert row in [0,1], 'row must be 0 or 1, found {}'.format(row)
        assert visual in [0,1], 'visual must be 0 or 1, found {}'.format(visual)
        
        nrows, ncols, _ = self._image.shape
        mask = self.minimum_seam(row)

        if visual != 0:
            tmp_image = np.copy(self._image)
            tmp_image[mask[:,:,0], 0] = 255
            tmp_image[mask[:,:,1], 1] = 0
            tmp_image[mask[:,:,2], 2] = 0
            plt.figure(figsize = (10,10))
            plt.imshow(tmp_image, cmap = 'gray')
            plt.tight_layout()
            plt.show()
            return tmp_image
        else:
            if row == 0:
                self._image = self._image[~mask].reshape(nrows, ncols-1, 3)
            else:
                self._image = np.rot90(self._image, 1)
                mask = np.rot90(mask, 1)
                self._image = self._image[~mask].reshape(ncols, nrows-1, 3)
                self._image = np.rot90(self._image, 3)
            return self._image
        

    ################################
    # DELETE MULTIPLE VERTICAL SEAM
    ################################

    def crop_column(self, remain = 0.8):
        '''Crop image along column.
        
        Parameter:
            remain(default 80%): if 0<remain<1 then remain is percentage of columns remaining after crop
                                 if remain > 1 then remain is number of columns remaining after crop
        
        Return:
            image after crop
        '''
        nrows, ncols, _ = self._image.shape
        assert remain > 0, 'remain must be greater than 0'
        assert (type(remain) == float and 0 < remain < 1) or type(remain) == int, 'remain must be integer or in range (0,1)'
        assert 0 < remain < 1 or 1 <= remain < ncols, \
                'number of column remain must be smaller than {0}, found {1}'.format(ncols, remain)

        if remain < 1:
            new_ncols = int(remain * ncols)
        else:
            new_ncols = remain

        for i in trange(ncols - new_ncols):
            self._image = self.seam_column_delete()

        return self._image 


    ################################
    # DELETE MULTIPLE HORIZONTAL SEAM
    ################################

    def crop_row(self, remain = 0.8):
        '''Crop image along row.
        
        Parameter:
            remain(default 80%): if 0<remain<1 then remain is percentage of rows remaining after crop
                                 if remain > 1 then remain is number of rows remaining after crop
        
        Return:
            image after crop
        '''
        nrows, ncols, _ = self._image.shape
        assert remain > 0, 'remain must be greater than 0'
        assert (type(remain) == float and 0 < remain < 1) or type(remain) == int, 'remain must be integer or in range (0,1)'
        assert 0 < remain < 1 or 1 <= remain < nrows, \
                'number of row remain must be smaller than {0}, found {1}'.format(nrows, remain)
        self._image = np.rot90(self._image, 1)
        self._image = self.crop_column(remain = remain)
        self._image = np.rot90(self._image, 3)
        return self._image


    ################################
    # RESTORE ORIGINAL IMAGE
    ################################

    def restore_image(self):
        '''Restore original image before any editing'''
        self._image = self._origin_image.copy()
        return self._image 


    ################################
    # CROP BOTH AXIS
    ################################

    def crop(self, row_remain = 0.8, col_remain = 0.8):
        '''Crop both row and column of image
        
        Parameter:
            row_remain(default 80%): int -> row remaining after crop
                                     float (0, 1) -> % row remaining after crop
            col_remain(default 80%): int -> column remaining after crop
                                     float (0, 1) ->% column remaining after crop
        '''
        nrows, ncols, _ = self._image.shape

        assert row_remain > 0, 'row_remain must be greater than 0'
        assert (type(row_remain) == float and 0 < row_remain < 1) or type(row_remain) == int, 'row_remain must be integer or in range (0,1)'
        assert 0 < row_remain < 1 or 1 <= row_remain < nrows, \
                'number of row remain must be smaller than {0}, found {1}'.format(nrows, row_remain)
        assert col_remain > 0, 'col_remain must be greater than 0'
        assert (type(col_remain) == float and 0 < col_remain < 1) or type(col_remain) == int, 'col_remain must be integer or in range (0,1)'
        assert 0 < col_remain < 1 or 1 <= col_remain < ncols, \
                'number of column remain must be smaller than {0}, found {1}'.format(ncols, col_remain)
        if nrows > ncols:
            self.crop_row(row_remain)
            self.crop_column(col_remain)
        else:
            self.crop_column(col_remain)
            self.crop_row(row_remain)
        return self._image 
    
        
    ################################
    # CROP TO SQUARE IMAGE
    ################################

    def crop_square(self, size = -1):
        ''' Crop image to square image

        Parameter:
            size: float (0,1) or integer > 1
                  size of image after crop. 
                  Default size = -1: 90% of smaller side of image
        
        Return: 
            square image
        '''
        nrows, ncols = self._image.shape[:2]

        assert size >= 0 or size == -1, 'size must be greater than 0'
        assert (type(size) == float and 0 < size < 1) or type(size) == int, 'size must be integer or in range (0,1)'
        assert size <= min(nrows, ncols), 'new size {0}x{0} must be smaller than size of image {1}x{2}'.format(size, nrows, ncols)

        if size == -1:
            size = int(0.9*min(nrows, ncols))
        if type(size) == float:
            size = int(size*min(nrows, ncols))

        self.crop(size, size)
        return self._image



if __name__ == '__main__':
    image_path = 'Images/image_4.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    check = 0
    sc = Seam_Carver(image)
    if check == 1:
        # sc.energy_image()
        # sc.minimum_seam(2)
        # sc.seam_column_delete(2,2)
        # sc.crop_column(1185)
        # sc.crop_row(671)
        # sc.crop(669, 1184)
        # sc.crop_square(size = 671)
        pass
    else:
        sc.energy_image()
        new_image = sc.crop_square(-1)

        plt.figure(figsize = (12, 8))
        plt.subplot(121)
        plt.imshow(image)
        plt.title('Original: ' + 'x'.join(map(str, list(image.shape))))
        plt.subplot(122)
        plt.imshow(new_image)
        plt.title('Resize: ' + 'x'.join(map(str, list(new_image.shape))))
        plt.tight_layout()
        plt.show()
        # for i in range(5):
        #     new_image = sc.seam_column_delete(row = 1)
        # plt.subplot(121)
        # plt.imshow(new_image)
        # plt.subplot(122)
        # plt.imshow(sc._image)
        # plt.show()
        pass
