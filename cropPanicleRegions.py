# crop the individual rachis from the original scan image
import os
import matplotlib.pyplot as plt
import glob 
from skimage import filters
from skimage import color
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from scipy import ndimage


path = "/home/ubuntu/picture/oringinalRachisImage"
out_path = "/home/ubuntu/picture/cropRachis" 
if not os.path.isdir(out_path):
        os.makedirs(out_path)

flist = sorted(glob.glob(path +'/*.jpg'))
# flist = sorted(glob.glob(path +'/*.png'))
for id in range(len(flist)):
    
    image = imread(flist[id])

    # file_name = os.path.splitext(flist[id])[0].split('\\')[-1]
    filename = os.path.basename(flist[id])
    out_sub_dir = os.path.join(out_path, filename[:3])
    # if not os.path.isdir(out_sub_dir):
    #      os.makedirs(out_sub_dir)

    # convert the image to grayscale
    gray_image = color.rgb2gray(image)
    blurred_image = filters.gaussian(gray_image, sigma=1.0)
    # blurred_image = filters.gaussian(image, sigma=1.0)
    thre = filters.threshold_otsu(blurred_image)
    bw_image = blurred_image < thre
    bw_image = ndimage.binary_fill_holes(bw_image)

    #bw_image = morphology.remove_small_objects(bw_image, 5000)
    label_image = label(bw_image)

    panicles = regionprops(label_image, blurred_image)
    sorted_panicles = sorted(panicles, key=lambda x: x.bbox[1])
    i = 0

    for region in sorted_panicles:
        if region.area > 10000 and region.minor_axis_length > 20:
            # out_file_path = os.path.join(out_sub_dir, '{}_{}.png'.format(filename[:-4], i))
            out_file_path = os.path.join(out_path, '{}_{}.png'.format(filename[:-4], i))
            # out_file_path = os.path.join(out_path, filename)
            #Bounding box (min_row, min_col, max_row, max_col).
            imsave(out_file_path, image[max(region.bbox[0]-50,20):min(region.bbox[2]+50, image.shape[0]), \
                max(region.bbox[1]-50,0):min(region.bbox[3]+50, image.shape[1])])
            i = i+1