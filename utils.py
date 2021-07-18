from PIL import Image, ImageOps
import numpy as np
import os
from random import sample
from keras.preprocessing.image import img_to_array, array_to_img

def load_img(file):
    """Load an image.

    Keyword arguments:
    file -- file to be loaded
    """
    
    img = Image.open(file, 'r')
    ext = file.split('.')[-1]

    if ext == 'png':
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    
    return img

def adjust_masks(masks_names, masks_dir, images_dir, masks_dir_processed):
    """Loads masks, resizes these to images sizes (if different) and saves adjusted masks into another directory.

    Keyword arguments:
    masks_names -- list of masks filenames
    masks_dir -- directory (path) containing the masks files
    images_dir -- directory containing the images (masks counterparts) files
    masks_dir_processed -- directory to save the adjusted masks
    """
    
    for file_mask_name in sorted(masks_names):
        file_image_name = file_mask_name.replace('png','jpg')
        img = load_img(os.path.join(images_dir, file_image_name))
        mask = load_img(os.path.join(masks_dir, file_mask_name))
        
        if mask.size != img.size:
            mask = mask.resize(img.size)

        mask.save(os.path.join(masks_dir_processed, file_mask_name), format='png')   

def crop_or_pad(img, size=640):
    """Loads an image and depending on its size, crops or pads it, and returns a squared image of size 640.

    Keyword arguments:
    img -- the image loaded
    size -- the size of the image output (default 640)
    """
    
    w, h = img.size[0], img.size[1]
    temp = Image.new(img.mode, (size, size), color=0)
            
    if w > size and h > size:
        temp.paste(img, (-int((w - size)/2),0)) # h starting in 0 to avoid cutting heads
        
    elif w > size and h <= size:
        temp.paste(img, (-int((w - size)/2),int((size - h)/2))) 
        
    elif w <= size and h > size:
        temp.paste(img, (int((size - w)/2),0)) # h starting in 0 to avoid cutting heads
        
    elif w <= size and h <= size:        
        temp.paste(img, (int((size - w)/2),int((size - h)/2))) 
                        
    return temp

def load_dataset(filenames, path):
    """Loads images from filenames in a path and returns a list of images and its mirrored images for augmentation.

    Keyword arguments:
    filenames -- the names of the image files to be loaded
    path -- the path where the files are
    """
    
    imgs = []
        
    for file in filenames:
        img = crop_or_pad(load_img(os.path.join(path, file)))
        imgs.append(img)
#         img = ImageOps.mirror(img) # augmenting by flipping image horizontally (left to right)
#         imgs.append(img)
        
    return imgs

def conv_imgs_to_arrays(imgs):
    """Converts a list of images into arrays scaling the elements by 1/255.

    Keyword arguments:
    imgs -- the list of images to be loaded and transformed
    """
    
    arrays = np.array(list(map(lambda img: img_to_array(img, dtype='float16'), imgs)), dtype='float16') / 255.
    
    return arrays

def conv_arrays_to_imgs(arrays):
    """Converts arrays into a list of images scaling the elements by 255.

    Keyword arguments:
    arrays -- the arrays to be loaded and transformed into images
    """
    
    imgs = list(map(lambda img: array_to_img(img, scale=255), arrays))
    
    return imgs