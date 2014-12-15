#!/usr/bin/env python
# encoding: utf-8
"""
Make segmentation tables corresponding to individual pixels in an image.

2014-12-04 - Created by Jonathan Sick
"""

from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
import numpy as np

from galaxycoords import galaxy_coords


class PixelSegmap(object):
    """Make a pixel segmentation table if each pixel its own segmentation."""
    def __init__(self, ref_header, flagmap, pixel_scale):
        super(PixelSegmap, self).__init__()
        self.ref_header = ref_header
        self.flagmap = flagmap
        self.ref_wcs = WCS(ref_header)
        self.pixel_scale = pixel_scale  # TODO get form image or WCS
        self.pixel_table = None
        self.segmap = None

    def segment(self, coord0, d0, incl0, pa0):
        """Segment galaxy image into wedges of `delta` opening angle.

        Radial grid should start > 0. Values between 0. and radial_grid[0]
        are assigned to a central elliptical bin.
        """
        self._map_pixel_coordinates(coord0, d0, incl0, pa0)
        self._make_pixel_table()
        self._make_segmap()

    def _map_pixel_coordinates(self, coord0, d0, incl0, pa0):
        shape = (self.ref_header['NAXIS2'], self.ref_header['NAXIS1'])
        print "ref shape", shape
        yindices, xindices = np.mgrid[0:shape[0], 0:shape[1]]
        self.y_indices_flat = yindices.flatten()
        self.x_indices_flat = xindices.flatten()
        self.ra, self.dec = self.ref_wcs.all_pix2world(self.x_indices_flat,
                                                       self.y_indices_flat, 0)
        coords = SkyCoord(self.ra, self.dec, "icrs", unit="deg")
        pixel_R, pixel_PA, sky_radius = galaxy_coords(coords,
                                                      coord0,
                                                      pa0,
                                                      incl0,
                                                      d0)
        self.pixel_R = pixel_R.kpc
        self.image_sky_r = sky_radius  # arcsec units
        self.image_pa = pixel_PA.deg

        # Measure position on the sky
        # TODO put into galaxy_coords?
        delta_ra = coords.ra - coord0.ra
        P = Angle(np.arctan2(np.sin(delta_ra.rad),
                             np.cos(coord0.dec.rad) * np.tan(coords.dec.rad)
                             - np.sin(coord0.dec.rad) * np.cos(delta_ra.rad)),
                  unit=u.rad)
        # Reset wrap-around range
        s = np.where(P < 0.)[0]
        P[s] = Angle(2. * np.pi, unit=u.rad) + P[s]
        P -= pa0
        s = np.where(P.deg < 0.)[0]
        P[s] += Angle(2. * np.pi, unit=u.rad)
        self.image_sky_pa = P.deg

    def _make_pixel_table(self):
        # filter out bad pixels
        print "flagmap shape", self.flagmap.shape
        s = np.where(self.flagmap.flatten() == 1)[0]
        pix_id = np.arange(len(s), dtype=np.int)
        area = np.ones(len(s), dtype=np.float) * self.pixel_scale
        t = Table((pix_id, self.x_indices_flat[s], self.y_indices_flat[s],
                   self.ra[s], self.dec[s],
                   self.image_sky_pa[s], self.image_pa[s],
                   self.pixel_R[s], self.image_sky_r[s],
                   area),
                  names=('ID', 'pixel_x', 'pixel_y',
                         'ra', 'dec',
                         'phi_sky', 'phi_disk',
                         "R_maj", "R_sky",
                         'area'))
        self.pixel_table = t

    def _make_segmap(self):
        shape = (self.ref_header['NAXIS2'], self.ref_header['NAXIS1'])
        self.segmap = np.empty(shape, dtype=np.int)
        self.segmap.fill(-1)
        x = self.pixel_table['pixel_x']
        y = self.pixel_table['pixel_y']
        self.segmap[y, x] = self.pixel_table['ID']
