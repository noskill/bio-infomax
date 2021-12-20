import pyvips
import itertools
import multiprocessing

image = pyvips.Image.new_from_file('/mnt/fileserver/shared/datasets/biodata/SCAN_09042015_100_LAME_1_1_201504091950.tif')

w = 68544
h = 74032
x = 68544/2
y = 74032/2
tile_size_w = 1000
tile_size_h = 1000


def write_crop_file_to_file(src_path, x, y, tile_size_w, tile_size_h, dst_path):
    image = pyvips.Image.new_from_file(src_path)
    tile = image.crop(x, y, tile_size_w, tile_size_h)
    buf = tile.write_to_file(dst_path)
    return buf


src_path = '/mnt/fileserver/shared/datasets/biodata/SCAN_09042015_100_LAME_1_1_201504091950.tif'
X = [x for x in range(0, w, tile_size_w)]
Y = [y for y in range(0, h, tile_size_h)]
out_path = '/mnt/fileserver/shared/datasets/biodata/TestImgOutputParallel/'
out_path_ = '/mnt/fileserver/shared/datasets/biodata/TestImgOutput/'


grid = [tuple([src_path])+g+(min(tile_size_w, w-g[0]), min(tile_size_h, h-g[1]), out_path + 'patch_{x}_{y}.jpg'.format(x=g[0], y=g[1])) for g in itertools.product(X, Y)]

# list(itertools.starmap(write_crop_file_to_file, grid))

pool = multiprocessing.Pool(4)
pool.starmap(write_crop_file_to_file, grid)
