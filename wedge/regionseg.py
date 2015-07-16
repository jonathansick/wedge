# encoding: utf-8
"""
Make segmentation maps where regions are defined by RA, Dec polygons.

2015-07-10 - Created by Jonathan Sick
"""

import numpy as np
from matplotlib.path import Path

# from astropy.wcs import WCS
from astropy.table import Table
from astropy.utils.console import ProgressBar

from .pixelseg import PixelSegmap


class RegionSegmap(PixelSegmap):
    """Make a pixel segmentation table with regions defined by polygons."""
    def __init__(self, ref_header, flagmap, pixel_scale):
        super(RegionSegmap, self).__init__(ref_header, flagmap, pixel_scale)

    def segment(self, regions, coord0, d0, incl0, pa0, metadata=None):
        """
        regions : list
            List of RA, Dec polygons, defined as (n, 2) numpy arrays.
        metadata : array
            Structured array of metadata associated with each region.
        """
        self._map_pixel_coordinates(coord0, d0, incl0, pa0)
        self._make_segmap(regions, metadata=metadata)

    def _make_segmap(self, regions, metadata=None):
        shape = (self.ref_header['NAXIS2'], self.ref_header['NAXIS1'])
        self.segmap = np.empty(shape, dtype=np.int)
        self.segmap.fill(-1)

        points = np.column_stack((self.ra, self.dec))

        dt = [('ID', int), ('ra', float), ('dec', float),
              ('phi_sky', float), ('phi_disk', float),
              ('R_maj', float), ('R_sky', float), ('area', float)]
        if metadata is not None:
            dt += [(n, t[0]) for n, t in metadata.dtype.fields.iteritems()]
        t = np.empty(len(regions), dtype=np.dtype(dt))

        with ProgressBar(len(regions)) as pbar:
            for i, region in enumerate(regions):
                region_path = Path(region)
                in_poly = region_path.contains_points(points)
                s = np.where(in_poly == True)[0]  # NOQA

                # Paint the segmap
                self.segmap[self.y_indices_flat[s], self.x_indices_flat[s]] = i

                area = float(len(s)) * self.pixel_scale ** 2.

                # Compute the mean centroid coordinate properties of each patch
                t['ID'][i] = i
                t['ra'][i] = self.ra[s].mean()
                t['dec'][i] = self.dec[s].mean()
                t['phi_sky'][i] = self.image_sky_pa[s].mean()
                t['phi_disk'][i] = self.image_pa[s].mean()
                t['R_maj'][i] = self.pixel_R[s].mean()
                t['R_sky'][i] = self.image_sky_r[s].mean()
                t['area'][i] = area
                if metadata is not None:
                    for n in metadata.dtype.names:
                        t[n][i] = metadata[n][i]
                pbar.update()

        self.pixel_table = Table(t)
