from __future__ import annotations

import numpy as np
from numpy.core.multiarray import ndarray

class NDVoxelArray( np.ndarray):

    ZAXIS, YAXIS, XAXIS = 0, 1, 2
    #
    AXIS_NAMES = ['z', 'y', 'x']

    _PLOT_KWARGS = {"origin": "lower"}

    def __new__(cls, input_array, voxdims=None, origin=None, info=None, **kwargs) -> NDVoxelArray:

        import time

        import copy

        if isinstance(input_array, np.ndarray):
            obj = input_array.view(NDVoxelArray)
        else:
            obj = np.asarray(input_array)
            obj = obj.view(NDVoxelArray)

        if voxdims is not None:
            assert (len(np.shape(obj)) == len(voxdims))
            obj.voxdims = tuple([float(s) for s in voxdims])
        else:
            # Default voxel size is 1
            obj.voxdims = np.ones(len(np.shape(input_array)))

        if origin is not None:
            assert (len(np.shape(obj)) == len(origin))
            obj.origin = origin

        else:
            # Default origin is 0,..
            obj.origin = np.zeros(len(np.shape(input_array)))

        if info is not None:
            obj.info = info
        else:
            obj.info = {}

        # bnds = [(obj.origin[s], obj.origin[s] + np.shape(obj)[s] * obj.voxdims[s]) for s in range(0, len(np.shape(obj)))]
        # obj._extent = [bnds[1][0],bnds[1][1],bnds[0][0],bnds[0][1]]

        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.voxdims = getattr(obj, 'voxdims', None)
        self.origin = getattr(obj, 'origin', None)
        self.info = getattr(obj, 'info', None)
        # print(self.origin,self.extent)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(NDVoxelArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.voxdims, self.origin, self.info)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.voxdims = state[-3]  # Set the info attribute
        self.origin = state[-2]  # Set the info attribute
        self.info = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(NDVoxelArray, self).__setstate__(state[0:-3])

    def __iadd__(self, other):
        if isinstance(other, NDVoxelArray) and len(other.shape) > 0:
            # print(other.voxdims, other.origin, other.shape)
            assert self.voxdims == other.voxdims
            assert self.origin == other.origin
            assert self.shape == other.shape
            return self + other
        else:
            return super(NDVoxelArray, self).__iadd__(other)

    def half_vox(self, dim):
        return self.voxdims[dim] / 2.0

    @staticmethod
    def from_ndvoxel(template, data):
        assert isinstance(template, NDVoxelArray)
        assert isinstance(data, ndarray)
        import copy
        return NDVoxelArray(data, voxdims=template.voxdims, origin=template.origin, info=copy.deepcopy(template.info))

    @property
    def PLOT_KWARGS(self):
        return {**self._PLOT_KWARGS, 'extent': self.extent}

    @property
    def bounds(self):
        return [(self.origin[s],
                 self.origin[s] + np.shape(self)[s] * self.voxdims[s])
                for s in range(0, len(np.shape(self)))]

    @property
    def center(self):
        return [(s[1] - s[0]) / 2.0 + s[0] for s in self.bounds]

    @property
    def extent(self):
        bnds = self.bounds
        return [bnds[1][0], bnds[1][1], bnds[0][0], bnds[0][1]]

    @property
    def nDim(self):
        return len(self.voxdims)

    @property
    def physicalDim(self):
        return self.evalDim(np.shape(self))

    @property
    def physicalVol(self):
        return np.prod(self.physicalDim)

    @property
    def voxel_edge_mesh(self):
        return tuple([np.linspace(self.origin[i],
                                  self.origin[i] + self.voxdims[i] * (self.shape[i]),
                                  self.shape[i] + 1)
                      for i in range(0, self.nDim)])

    @property
    def coordinate_mesh(self, dim=None) -> list[list[float]]:
        to_ret = []

        for i in range(0, self.nDim):
            to_ret.append(self.get_coordinate_mesh(i))
            # to_ret.append(np.linspace(self.origin[i]+self.half_vox(i),
            #                           self.origin[i]+self.voxdims[i]*self.shape[i]-self.half_vox(i),
            #                           self.shape[i], endpoint=True))
        return to_ret

    def get_coordinate_mesh(self, dim: int) -> list[float]:
        assert (dim < self.nDim)
        return np.linspace(self.origin[dim] + self.half_vox(dim),
                           self.origin[dim] + self.voxdims[dim] * self.shape[dim] - self.half_vox(dim),
                           self.shape[dim], endpoint=True)

    @property
    def coordinate_array(self) -> tuple[list[float]]:
        return np.meshgrid(*self.coordinate_mesh, indexing='ij')

    def evalDim(self, shapeTuple):
        return np.array(shapeTuple) * np.array(self.voxdims)

    def evalRectVolume(self, rectSizeTuple):
        return np.prod(self.evalDim(rectSizeTuple))

    def evalVoxValueVolume(self, value):
        return len(np.where(self == value)[0]) * np.prod(self.voxdims)

    def get_mid_point(self):
        pass

    def interp2d(self, kind='cubic'):
        from scipy import interpolate
        if self.nDim == 2:
            return interpolate.interp2d(self.coordinate_mesh[1], self.coordinate_mesh[0],
                                        self, kind=kind)
        else:
            raise NotImplementedError()

    def interp_to_NDVoxel(self, target_interp_vox: NDVoxelArray, interp_kind: str = 'cubic') -> NDVoxelArray:

        if self.nDim == 2 and target_interp_vox.nDim == 2:

            bs = self.bounds
            bt = target_interp_vox.bounds

            if not all(s[0] <= t[0] and s[1] >= t[1] for s, t in zip(bs, bt)):
                # print(bs, target_interp_vox.bounds)
                # raise NotImplementedError('target coord system must be within bounds of NDVoxel')
                pass

            int_data = self.interp2d(kind=interp_kind)(target_interp_vox.coordinate_mesh[1],
                                                       target_interp_vox.coordinate_mesh[0])

            return NDVoxelArray(int_data, origin=target_interp_vox.origin, voxdims=target_interp_vox.voxdims)

        else:
            raise NotImplementedError()

    def crop_to_NDVoxel(self, target_crop_vox: NDVoxelArray) -> NDVoxelArray:
        if self.nDim == 2 and target_crop_vox.nDim == 2:

            bs = self.bounds
            bt = target_crop_vox.bounds
            print(bt)

            if not all(s[0] <= t[0] and s[1] >= t[1] for s, t in zip(bs, bt)):
                raise NotImplementedError('target coord system must be within bounds of NDVoxel')

            return self.subset(bt)

        else:
            raise NotImplementedError()

    def resample(self, new_vox_dims, zoom_order=3) -> NDVoxelArray:

        zoom_factor = np.array(self.voxdims) / np.array(new_vox_dims)

        import scipy.ndimage
        new_eye_vol = scipy.ndimage.zoom(self, zoom_factor, order=zoom_order)

        return NDVoxelArray(new_eye_vol, voxdims=np.array(self.voxdims) / zoom_factor, origin=self.origin)

    # def resample2(self, new_vox_dims, zoom_order=3) -> NDVoxelArray:
    #
    #     zoom_factor = np.array(self.voxdims) / np.array(new_vox_dims)
    #
    #     import scipy.ndimage
    #     new_eye_vol = scipy.ndimage.zoom(self, zoom_factor, order=zoom_order)
    #
    #     return NDVoxelArray(new_eye_vol, voxdims=np.array(self.voxdims) / zoom_factor, origin=self.origin)

    def index_to_coord(self, index, dimension):
        assert dimension >= 0 and dimension < len(self.shape)
        assert index >= 0 and index < self.shape[dimension]
        return self.origin[dimension] + (index + 0.5) * self.voxdims[dimension]

    def index_tuple_to_coord(self, index_list):
        assert len(index_list) == len(self.shape)
        return [self.index_to_coord(cent, ind) for ind, cent in enumerate(index_list)]

    def center_index(self):
        return [int(s / 2.0) for s in self.shape]

    def real_coord(self, index, dimension):

        assert index <= self.shape[dimension] and index >= 0
        return self.origin[dimension] + (index * self.voxdims[dimension])

    def coord_to_nearest_index(self, coord, dimension, do_clipping=False):
        """
        :param coord:
        :param dimension:
        :param do_clipping: if true, and subset is larger than parent array, return maximum size subset
        :return:
        """
        assert dimension >= 0 and dimension < len(self.shape)

        # if coordinate outside of array, return maximum / minimum of array coords
        if do_clipping:
            if coord < self.origin[dimension]:
                coord = self.origin[dimension]
            if coord > self.bounds[dimension][1]:
                coord = self.bounds[dimension][1]
        else:
            assert coord >= self.origin[dimension] and coord <= self.bounds[dimension][1]

        return int(np.floor((coord - self.origin[dimension]) / self.voxdims[dimension]))

    def fraction_to_coord(self, fraction, dimension):
        '''
        Returns coordinate value for given fraction from lower edge of image
        :param fraction: decimal on 0->1 representing fraction of distance from edge of image
        :param dimension: index of desired coordinate (z,y,x) = (0,1,2)
        :return: coordinate representation of distance from edge
        '''
        assert (0.0 <= fraction <= 1.0)
        return self.origin[dimension] + fraction * self.voxdims[dimension] * self.shape[dimension]

    def fraction_to_nearest_index(self, fraction, dimension, do_clipping=False):
        '''
        Returns nearest index for given fraction from lower edge of image
        :param fraction: decimal on 0->1 representing fraction of distance from edge of image
        :param dimension: index of desired coordinate (z,y,x) = (0,1,2)
        :return: coordinate representation of distance from edge
        '''
        assert (0.0 <= fraction <= 1.0)
        return self.coord_to_nearest_index(self.fraction_to_coord(fraction, dimension),
                                           dimension, do_clipping=do_clipping)

    def coord_tuple_to_nearest_index(self, coord_list):
        assert len(coord_list) == len(self.shape)
        return [self.coord_to_nearest_index(coord, ind) for ind, coord in enumerate(coord_list)]

    def npy_index_list_to_coords(self, index_list):
        '''

        :param index_list: List of array indeces of the form [[z1,z2,...],[y1,y2..],[x1,x2...]]
        :return: list of coordinate of the same form
        '''

        assert (len(index_list) == self.nDim)

        nindex = len(index_list[0])

        #  make sure all index lists are same size
        assert (all([len(s) == nindex for s in index_list]))

        return [[self.index_tuple_to_coord(index_list[i][j]) for j in range(0, nindex)] for i in range(0, self.nDim)]

    # def subset(self, coord_bounds: list, do_clipping: bool = False, buffer_pix: int = 0):
    #     '''
    #     Generate subset of voxel array
    #     :param coord_bounds: list of tuples of the form [ (low,high), ..., len(self.shape)]
    #     :param do_clipping: if true, and subset is larger than parent array, return maximum size subset
    #     :return: sub image defined by low,high coordinates
    #     '''
    #
    #     assert len(coord_bounds) == len(self.shape)
    #
    #     for l, h in coord_bounds:
    #         assert l < h
    #
    #     this_origin = []
    #     ind_lims = []
    #
    #
    #
    #     for d in range(0, len(coord_bounds)):
    #         l, h = coord_bounds[d]
    #
    #         l = l - buffer_pix
    #         h = h + buffer_pix
    #
    #
    #
    #         li = self.coord_to_nearest_index(l, d, do_clipping=do_clipping)
    #
    #         lo = self.index_to_coord(li, d) - (self.voxdims[d] / 2.0)
    #
    #
    #
    #         nv = int(np.ceil((h - l) / self.voxdims[d]))
    #
    #         # throw error if a larger than array subset is requested
    #         if li + nv > self.shape[d] and not do_clipping:
    #             raise RuntimeError('requested subset of %s voxels in dimension %s is larger than parent array dimension %s'%(li+nv,d,self.shape[d]))
    #
    #         ind_lims = ind_lims + [slice(li, li + nv)]
    #         this_origin = this_origin + [lo]
    #
    #     return NDVoxelArray(self[tuple(ind_lims)], voxdims=self.voxdims, origin=this_origin) # type: NDVoxelArray

    def subset(self, coord_bounds: list, do_clipping: bool = False):
        '''
        Generate subset of voxel array
        :param coord_bounds: list of tuples of the form [ (low,high), ..., len(self.shape)]
        :param do_clipping: if true, and subset is larger than parent array, return maximum size subset
        :return: sub image defined by low,high coordinates
        '''

        assert len(coord_bounds) == len(self.shape)
        # for l, h in coord_bounds:
        #     assert l < h

        ind_lims = []

        for d in range(0, len(coord_bounds)):

            try:
                l, h = coord_bounds[d]
            except TypeError:
                l = coord_bounds[d]
                h = coord_bounds[d]

            li = self.coord_to_nearest_index(l, d, do_clipping=do_clipping)

            nv = int(np.ceil((h - l) / self.voxdims[d]))

            # throw error if a larger than array subset is requested
            if li + nv > self.shape[d] and not do_clipping:
                raise RuntimeError(
                    'requested subset of %s voxels in dimension %s is larger than parent array dimension %s' % (
                    li + nv, d, self.shape[d]))

            ind_lims = ind_lims + [(li, li + nv)]

        return self.subset_by_index(ind_lims, do_clipping=do_clipping)


    # def coords_from_image_fraction(self):

    def subset_by_index(self, coord_bounds: list, do_clipping: bool = False):
        '''
        Generate subset of voxel array
        :param coord_bounds: list of tuples of the form [ (low,high), ..., len(self.shape)]
        :param do_clipping: if true, and subset is larger than parent array, return maximum size subset
        :return: sub image defined by low,high coordinates # type: NDVoxelArray
        '''

        assert len(coord_bounds) == len(self.shape)

        this_origin = []
        ind_lims = []
        voxdim_lims = []

        for d in range(0, len(coord_bounds)):

            try:
                l, h = coord_bounds[d]
            except TypeError:
                if isinstance(coord_bounds[d], int):
                    l = coord_bounds[d]
                    h = coord_bounds[d]
                else:
                    raise NotImplementedError()

            assert l <= h

            li = int(l)

            lo = self.index_to_coord(li, d) - (self.voxdims[d] / 2.0)

            nv = int(h - l)

            # throw error if a larger than array subset is requested
            if li + nv > self.shape[d] and not do_clipping:
                raise RuntimeError(
                    'requested subset of %s voxels in dimension %s is larger than parent array dimension %s' % (
                    li + nv, d, self.shape[d]))

            if nv > 1:
                ind_lims = ind_lims + [slice(li, li + nv)]
                voxdim_lims = voxdim_lims + [self.voxdims[d]]
                this_origin = this_origin + [lo]
            else:
                ind_lims = ind_lims + [li]

        return NDVoxelArray(self[tuple(ind_lims)], voxdims=voxdim_lims, origin=this_origin)  # type: NDVoxelArray

    def slice_3D_to_2D(self, slice_dim, slice_index, do_clipping=False) -> NDVoxelArray:

        if not self.nDim == 3:
            raise NotImplementedError("slice_3D_to_2D only implemented for 3D NDVoxelArray")

        coord_bounds = [[0, self.shape[s]] for s in range(0, self.nDim)]
        coord_bounds[slice_dim] = slice_index

        return self.subset_by_index(coord_bounds, do_clipping=do_clipping)

    def to_dataframe_dict(self):
        # return {'input_array': self.view(np.array), 'voxdims': self.voxdims, 'origin': self.origin, 'info': None}
        return {'input_array': self.tolist(), 'voxdims': self.voxdims, 'origin': self.origin, 'info': None}

    @classmethod
    def from_dataframe_dict(cls, df_dict):
        # print("from_dataframe_dict")

        toret = NDVoxelArray(df_dict['input_array'], voxdims=df_dict['voxdims'], origin=df_dict['origin'],
                             info=df_dict['info'])
        return toret

    @property
    def stats(self) -> str:
        retstr = "%s: %s\n" \
                 "\tvox_size: %s\n" \
                 "\torigin: %s\n" \
                 "\tsize: %s" % (self.__class__.__name__, self.shape, self.voxdims, self.origin, self.physicalDim)
        return retstr

class FilmImage(object):

    def __init__(self, image_file, dpi):
        '''
        store film image as optical density normalized by flood image
        :param image_file: image file name
        '''
        import os
        assert(os.path.splitext(image_file)[1] == '.tif')
        self.file_name = image_file
        self.reso_mm = (1.0 / dpi) * 25.4

        from tifffile import TiffFile
        film_image_tiff = TiffFile(self.file_name)
        if len(film_image_tiff.pages) != 1:
            raise NotImplementedError('only TIF files with 1 page are implemented')

        from tifffile import TiffFile
        self.data = film_image_tiff.pages[0].asarray()

    @property
    def red_channel(self):
        return self.data[:,:,0]

    @property
    def green_channel(self):
        return self.data[:,:,1]

    @property
    def blue_channel(self):
        return self.data[:,:,2]

import os
def list_files_by_wildcard(pattern, recursive=False, do_sort=True):
    """
    List files matching UNIX wildcard pattern
    :param pattern: unix wildcard pattern
    :param recursive: if true, recurse into directory
    """
    import fnmatch
    _dirname, wc = os.path.split(pattern)

    if recursive:

        toret = [os.path.join(dirPath, f) for dirPath, subdirNames, filenames in os.walk(_dirname)
                 for f in filenames if fnmatch.fnmatch(os.path.join(dirPath, f), wc)]

    else:
        toret = [os.path.join(_dirname, f) for f in os.listdir(_dirname) if fnmatch.fnmatch(f, wc)]

    if do_sort:
        return sorted(toret)
    else:
        return toret

