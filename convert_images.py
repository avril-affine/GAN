import os
import sys
from PIL import Image


def main():
    if len(sys.argv) != 4:
        print 'Wrong number of inputs'
        return

    image_dir = sys.argv[2]
    output_dir = sys.argv[3]

    dim = int(sys.argv[1])
    out_size = (dim, dim)

    filenames = [x for x in os.listdir(image_dir) if not x.startswith('.')]
    for image_name in filenames:
        filename = os.path.join(image_dir, image_name)
        img = Image.open(filename)
        # img.thumbnail(out_size, Image.ANTIALIAS)
        img = img.resize(out_size, Image.BILINEAR)
        out_file = os.path.join(output_dir, image_name)
        img.save(out_file, "JPEG")


if __name__ == '__main__':
    main()
