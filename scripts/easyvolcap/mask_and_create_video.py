from os.path import join
from glob import glob
import numpy as np
import os
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import save_image, load_image, generate_video


@catch_throw
def main():

    ground_mask = 'ground_mask.png'
    images = 'images'
    masks = 'masks'
    output = 'output'

    imgs = glob(join(images, '*'))
    msks = [i.replace(images, masks) for i in imgs]
    if len(msks) and not os.path.exists(msks[0]):
        msks = [i.replace('.jpg', '.png') for i in msks]

    grd = load_image(ground_mask)[..., -1]  # only the last channel, in 0, 1
    for img, msk in tqdm(zip(imgs, msks), total=len(imgs)):
        im = load_image(img)
        mk = load_image(msk)
        mk = np.clip(mk + grd, 0, 1)
        save_image(img.replace(images, output).replace('.jpg', '.png'), np.concatenate([im, mk[..., None]], axis=-1))

    # generate_video(f'"{output}/*"', output + '.mp4', 30)

if __name__ == '__main__':
    main()
