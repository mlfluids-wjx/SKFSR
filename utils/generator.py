import numpy as np

def sierpinski_foam(dmin=1, n=5, ndim=2, max_size=1e9):
    def _insert_cubes(im, n):
        if n > 0:
            n -= 1
            shape = np.asarray(np.shape(im))
            im = np.tile(im, (3, 3, 3))
            im[shape[0]:2*shape[0], shape[1]:2*shape[1], shape[2]:2*shape[2]] = 0
            if im.size < max_size:
                im = _insert_cubes(im, n)
        return im

    def _insert_squares(im, n):
        if n > 0:
            n -= 1
            shape = np.asarray(np.shape(im))
            im = np.tile(im, (3, 3))
            im[shape[0]:2*shape[0], shape[1]:2*shape[1]] = 0
            if im.size < max_size:
                im = _insert_squares(im, n)
        return im

    im = np.ones([dmin]*ndim, dtype=int)
    if ndim == 2:
        im = _insert_squares(im, n)
    elif ndim == 3:
        im = _insert_cubes(im, n)
        
    return im


