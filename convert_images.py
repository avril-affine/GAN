import os
from PIL import Image


if __name__ == '__main__':
    image_dir = 'flickr'
    output_dir = 'flickr_resize'
    out_size = (128, 128)

    filenames = [x for x in os.listdir(image_dir) if not x.startswith('.')]
    for image_name in filenames:
        filename = os.path.join(image_dir, image_name)
        img = Image.open(filename)
        # img.thumbnail(out_size, Image.ANTIALIAS)
        img = img.resize(out_size, Image.BILINEAR)
        out_file = os.path.join(output_dir, image_name)
        img.save(out_file, "JPEG")
