from io import BytesIO

import numpy as np
from PIL import Image


def png_to_jpg(img, quality):
    # check if the img in right
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    out = BytesIO()
    # ranging from 0-95, 75 is default
    img.save(out, format="jpeg", quality=quality)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()

    return img
