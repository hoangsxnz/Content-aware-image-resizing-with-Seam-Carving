import numpy as np 
import os
import cv2 
import matplotlib.pyplot as plt
from Seam_Carver import Seam_Carver
from tqdm import trange
from PIL import Image


def read_image(enter = 0, image_path = 'Images/image_1.jpg'):
    if enter == 1:
        print('Enter file name: ')
        image_path = input()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_path


def energy_visualize(image):
    sc = Seam_Carver(image)
    sc.energy_image(visual = 1)


def energy_image(image, filename):
    sc = Seam_Carver(image)
    energy = sc.energy_image()
    cv2.imwrite(filename.split('.')[0]+'_energy.jpg', cv2.normalize(energy, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))


def crop_image(image, filename, row = 0.8, col = 0.8):
    sc = Seam_Carver(image)
    crop = sc.crop(row, col)
    cv2.imwrite(filename.split('.')[0]+'_crop.jpg', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print('Crop finished: {} to {}'.format(image.shape, crop.shape))


def seam_visualize(image, axis = 0):
    # axis = 0 ---> column
    # axis = 1 ---> row 
    sc = Seam_Carver(image)
    sc.seam_column_delete(visual = 1, row = axis)


def column_visualize(image, col, filename):
    sc = Seam_Carver(image)
    nrows, ncols, _ = image.shape
    images = np.zeros((col, nrows, ncols, 3), dtype = np.int)
    for i in trange(col):
        mask = sc.minimum_seam()
        tmp_image = sc._image.copy()
        tmp_image[mask[:,:,0], 0] = 255
        tmp_image[mask[:,:,1], 1] = 0
        tmp_image[mask[:,:,2], 2] = 0
        sc.seam_column_delete()
        images[i, :, :ncols-i, :] = tmp_image
    imgs = [Image.fromarray(images[i].astype(np.uint8)) for i in range(col)]
    path = filename.split('.')[0] + '_col_process.gif'
    imgs[0].save(path, save_all=True, interlace = True, append_images=imgs[1:], duration=200, loop=0)
    Image.fromarray(sc._image).save(filename.split('.')[0]+'_col_result.jpg')
    print('Delete {} vertical seam: from {} to {}'.format(col, image.shape, sc._image.shape))
    return sc._image


def row_visualize(image, row, filename):
    sc = Seam_Carver(image)
    nrows, ncols, _ = image.shape
    images = np.zeros((row, nrows, ncols, 3), dtype = np.int)
    for i in trange(row):
        mask = sc.minimum_seam(1)
        tmp_image = sc._image.copy()
        tmp_image[mask[:,:,0], 0] = 255
        tmp_image[mask[:,:,1], 1] = 0
        tmp_image[mask[:,:,2], 2] = 0
        sc.seam_column_delete(row = 1)
        images[i, i:, :, :] = tmp_image
    imgs = [Image.fromarray(images[i].astype(np.uint8)) for i in range(row)]
    path = filename.split('.')[0] + '_row_process.gif'
    imgs[0].save(path, save_all=True, interlace = True, append_images=imgs[1:], duration=200, loop=0)
    Image.fromarray(sc._image).save(filename.split('.')[0]+'_row_result.jpg')
    print('Delete {} horizontal seam: from {} to {}'.format(row, image.shape, sc._image.shape))
    return sc._image


def crop_image(image, row, col, filename):
    col_image = column_visualize(image, col, filename)
    final_image = row_visualize(col_image, row, filename)


def input_percent(s):
    while True:
        try:
            print(s, end = '')
            percent = float(input())
            assert 0 < percent < 1, 'Invalid percent'
            break 
        except AssertionError as e:
            print(e)
            print('Enter again')
        except ValueError:
            print('Input an float in range(0, 1)')    
    return percent 


def input_path():
    while True:
        try:
            image, path = read_image(enter=1)
            break 
        except Exception as e:
            # print(e)
            print('Image not found!')
            print('Enter again(Y/N)')
            if input().lower() == 'n':
                os._exit(1)
    return image, path 


if __name__ == '__main__':
    while True:
        image, path = input_path()
        # path = 'image_1.jpg'
        # image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        nrows, ncols, _ = image.shape
        print('=====================')
        print('Image size {}x{}x{}'.format(nrows, ncols, 3))
        print('=====================')
        print('Option: ')
        print('1. Crop image along row')
        print('2. Crop image along column')
        print('3. Crop image both side')
        print('4. Energy image')
        print('5. Exit')
        print('======================')
        while True:
            choice = int(input('Choose: '))
            if not 1 <= choice <= 5:
                print('Choice invalid\nEnter again')
                continue
            else:
                break 
        if choice == 1:
            row = input_percent('Enter %% row remaining: ')
            num_del = int(nrows*(1-row))
            row_visualize(image, num_del, path)

        elif choice == 2:
            col = input_percent('Enter %% column remaining: ')
            num_del = int(ncols*(1-col))
            column_visualize(image, num_del, path)

        elif choice == 3:
            row = input_percent('Enter %% row remaining: ')
            col = input_percent('Enter %% column remaining: ')
            row_del = int(nrows*(1-row))
            col_del = int(ncols*(1-col))
            crop_image(image, row_del, col_del, path)

        elif choice == 4:
            energy_image(image, path)

        elif choice == 5:
            os._exit(1)

        out = input('Continue?(y/n)')
        if out.lower() == 'n':
            break

        
            
        



