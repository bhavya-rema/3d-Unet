import sys  
print(sys.path)

import numpy as np
import keras
import keras.models
from utils import tversky_crossentropy, dice_coefficient
import nibabel as nib
# import tensorflow as tf
import math


class Unet3D():
    def __init__(self):

        self.model_name = '3D-Unet.h5'
        super().__init__()
        self.window = (128, 128, 32)
        self.overlap = (0, 0, 0)

    def load_model(self, file_path=None):
        if file_path is not None:
            self.model = keras.models.load_model(file_path,
                                                 custom_objects={'tversky_crossentropy': tversky_crossentropy,
                                                                 'dice_coefficient': dice_coefficient})
        else:
            self.model = keras.models.load_model(self.model_name,
                                                 custom_objects={'tversky_crossentropy': tversky_crossentropy,
                                                                 'dice_coefficient': dice_coefficient})

    def make_patches(self, img):
        steps_x = int(math.ceil((len(img) - self.overlap[0]) /
                                float(self.window[0] - self.overlap[0])))
        steps_y = int(math.ceil((len(img[0]) - self.overlap[1]) /
                                float(self.window[1] - self.overlap[1])))
        steps_z = int(math.ceil((len(img[0][0]) - self.overlap[2]) /
                                float(self.window[2] - self.overlap[2])))
        # print(steps_x, steps_y, steps_z)
        patches = []
        for x in range(0, steps_x):
            for y in range(0, steps_y):
                for z in range(0, steps_z):
                    # Define window edges
                    x_start = x * self.window[0] - x * self.overlap[0]
                    x_end = x_start + self.window[0]
                    y_start = y * self.window[1] - y * self.overlap[1]
                    y_end = y_start + self.window[1]
                    z_start = z * self.window[2] - z * self.overlap[2]
                    z_end = z_start + self.window[2]
                    # Adjust ends
                    if (x_end > len(img)):
                        # Create an overlapping patch for the last images / edges
                        # to ensure the fixed patch/window sizes
                        x_start = len(img) - self.window[0]
                        x_end = len(img)
                        if x_start < 0: x_start = 0
                    if (y_end > len(img[0])):
                        y_start = len(img[0]) - self.window[1]
                        y_end = len(img[0])
                        if y_start < 0: y_start = 0
                    if (z_end > len(img[0][0])):
                        z_start = len(img[0][0]) - self.window[2]
                        z_end = len(img[0][0])
                        if z_start < 0: z_start = 0
                    # Cut window
                    window_cut = img[x_start:x_end, y_start:y_end, z_start:z_end]
                    # Add to result list
                    patches.append(window_cut)
        return patches

    def make_images(self, img,shape,m_patches):
        steps_x = int(math.ceil((len(img) - self.overlap[0]) /
                                float(self.window[0] - self.overlap[0])))
        steps_y = int(math.ceil((len(img[0]) - self.overlap[1]) /
                                float(self.window[1] - self.overlap[1])))
        steps_z = int(math.ceil((len(img[0][0]) - self.overlap[2]) /
                                float(self.window[2] - self.overlap[2])))
        patch_index = 0
        output_mask = np.zeros(shape)
        for x in range(0, steps_x):
            for y in range(0, steps_y):
                for z in range(0, steps_z):
                    # Define window edges
                    x_start = x * self.window[0] - x * self.overlap[0]
                    x_end = x_start + self.window[0]
                    y_start = y * self.window[1] - y * self.overlap[1]
                    y_end = y_start + self.window[1]
                    z_start = z * self.window[2] - z * self.overlap[2]
                    z_end = z_start + self.window[2]
                    # Adjust ends
                    if (x_end > len(img)):
                        # Create an overlapping patch for the last images / edges
                        # to ensure the fixed patch/window sizes
                        x_start = len(img) - self.window[0]
                        x_end = len(img)
                        if x_start < 0: x_start = 0
                    if (y_end > len(img[0])):
                        y_start = len(img[0]) - self.window[1]
                        y_end = len(img[0])
                        if y_start < 0: y_start = 0
                    if (z_end > len(img[0][0])):
                        z_start = len(img[0][0]) - self.window[2]
                        z_end = len(img[0][0])
                        if z_start < 0: z_start = 0
                    if patch_index <= len(m_patches)-1 and m_patches[patch_index].any() > 0:
                        output_mask[x_start:x_end, y_start:y_end, z_start:z_end] = m_patches[patch_index]
                    patch_index += 1
        return output_mask

    def predict(self, image_path):
        filename = image_path.split('/').pop().split('\\').pop()
        HU_MIN = -1000
        HU_MAX = 500
        HU_RANGE = HU_MAX - HU_MIN
        model = self.model
        ct_img = nib.load(image_path).get_fdata()
        #         ct_img = NiiImage(image_path)
        ct_shape = ct_img.shape
        image = np.reshape(ct_img, ct_shape + (1,))
        print(np.min(ct_img), np.max(ct_img))
        for x in range(len(ct_img)):
            for y in range(len(ct_img[x])):
                for z in range(len(ct_img[x, y])):
                    v = ct_img[x, y, z]
                    if filename.startswith('coronacases'):
                        if v < HU_MIN:
                            v = HU_MIN
                        elif v > HU_MAX:
                            v = HU_MAX
                        ct_img[x, y, z] = (v - HU_MIN) * 1 / HU_RANGE
                    else:
                        ct_img[x, y, z] = v / 255
        result_batches = []
        mask_patches = []
        ct_patches = self.make_patches(image)
        print(len(ct_patches))
        batch_img = []
        batch_count = 1
        img_data = []
        for i, patch in enumerate(ct_patches):
            batch_img.append(patch)
            batch_count += 1
            if batch_count > 4:
                batch_count = 1
                batch = np.array(batch_img)
                batch_img = []
                img_data.append(batch)
        for batch_item in img_data:
            result = model.predict(batch_item)
            result_batches.append(result)
        for batch in result_batches:
            b_masks= np.argmax(batch, axis=-1)
            for mask in b_masks:
                print(np.unique(mask))
                mask_patches.append(mask)
        print(len(mask_patches))
        original = self.make_images(image, ct_shape, mask_patches)

        return original
