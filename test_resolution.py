from model import resolve_single
from model.edsr import edsr
from model.srgan import generator

from utils import load_image
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_data", default="0829x4-crop.png", help="input-test-image")

args = parser.parse_args()

# load edsr model
# edsr_model = edsr(scale=4, num_res_blocks=16)
# edsr_model.summary()
# edsr_model.load_weights('weights/edsr-16-x4/weights.h5')
#
# load srgan model
srgan_model = generator()
srgan_model.load_weights('weights/srgan/gen_generator.h5')
srgan_model.summary()



# load picture
file = args.test_data
lr = load_image(f'demo/{file}')
print(lr.shape)
# hr_edsr = resolve_single(edsr_model, lr)
t = time.time()
hr_srgan = resolve_single(srgan_model, lr)
tt = time.time()

print(tt-t)

if not os.path.exists('./results'):
    os.mkdir('./results')

img_lr = Image.fromarray(lr, 'RGB')
# im_edsr = Image.fromarray(np.asarray(hr_edsr), 'RGB')
im_srgan = Image.fromarray(np.asarray(hr_srgan), 'RGB')

img_lr.save(f'./results/lr_{file}')
# img_resize = img_lr.resize((320*4, 240*4), Image.BICUBIC)
# img_resize.save(f'./results/resize_{file}')
# im_edsr.save(f'./results/edsr_{file}')
im_srgan.save(f'./results/srgan_{file}')
#
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(lr)
axes[0].set_title('LR')
# axes[1].imshow(hr_edsr)
# axes[1].set_title('SR_EDSR(x4)')
axes[1].imshow(hr_srgan)
axes[1].set_title('SR_SRGAN(x4)')
plt.tight_layout()
plt.show()