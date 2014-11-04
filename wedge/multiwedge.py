#!/usr/bin/env python
# encoding: utf-8
"""
Pipeline for dividing an image of an inclined disk galaxy into wedges that
cover the entire galaxy disk.

2014-11-03 - Created by Jonathan Sick
"""

# import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import Distance, Angle, SkyCoord
from astropy import units as u
import numpy as np


class MultiWedge(object):
    """Segment a galaxy image into multiple wedges."""
    def __init__(self, ref_header, seg_image=None, pixel_table=None):
        super(MultiWedge, self).__init__()
        self.ref_header = ref_header
        self.ref_wcs = WCS(ref_header)
        self.seg_image = seg_image
        self.pixel_table = pixel_table

    def segment(self, coord0, d0, incl0, pa0, pa_delta, radial_grid):
        """Segment galaxy image into wedges of `delta` opening angle.
        """
        self._map_pixel_coordinates(coord0, d0, incl0, pa0)

    def _map_pixel_coordinates(self, coord0, d0, incl0, pa0):
        shape = (self.ref_header['NAXIS2'], self.ref_header['NAXIS2'])
        self.seg_image = -1. * np.ones(shape, dtype=np.int)
        yindices, xindices = np.mgrid[0:shape[0], 0:shape[1]]
        y_indices_flat = yindices.flatten()
        x_indices_flat = xindices.flatten()
        ra, dec = self.ref_wcs.all_pix2world(x_indices_flat, y_indices_flat, 0)
        coords = SkyCoord(ra, dec, "icrs", unit="deg")
        pixel_R, pixel_PA = self.correct_rgc(coords,
                                             coord0,
                                             pa0,
                                             incl0,
                                             d0)
        self.image_R = pixel_R.kpc.reshape(*shape)
        self.image_PA = pixel_PA.rad.reshape(*shape)

    @staticmethod
    def correct_rgc(coord, glx_ctr, glx_PA, glx_incl, glx_dist):
        """Computes deprojected galactocentric distance.

        Inspired by: http://idl-moustakas.googlecode.com/svn-history/
            r560/trunk/impro/hiiregions/im_hiiregion_deproject.pro

        Parameters
        ----------
        coord : :class:`astropy.coordinates.SkyCoord`
            Coordinate of points to compute galactocentric distance for.
            Can be either a single coordinate, or array of coordinates.
        glx_ctr : :class:`astropy.coordinates.SkyCoord`
            Galaxy center.
        glx_PA : :class:`astropy.coordinates.Angle`
            Position angle of galaxy disk.
        glx_incl : :class:`astropy.coordinates.Angle`
            Inclination angle of the galaxy disk.
        glx_dist : :class:`astropy.coordinates.Distance`
            Distance to galaxy.

        Returns
        -------
        obj_dist : class:`astropy.coordinates.Distance`
            Galactocentric distance(s) for coordinate point(s).
        """
        # distance from coord to glx centre
        sky_radius = glx_ctr.separation(coord)
        avg_dec = 0.5 * (glx_ctr.dec + coord.dec).radian
        x = (glx_ctr.ra - coord.ra) * np.cos(avg_dec)
        y = glx_ctr.dec - coord.dec
        # azimuthal angle from coord to glx  -- not completely happy with this
        phi = glx_PA - Angle('90d') \
            + Angle(np.arctan2(y.arcsec, x.arcsec), unit=u.rad)

        # convert to coordinates in rotated frame, where y-axis is galaxy major
        # ax; have to convert to arcmin b/c can't do sqrt(x^2+y^2) when x and y
        # are angles
        xp = (sky_radius * np.cos(phi.radian)).arcmin
        yp = (sky_radius * np.sin(phi.radian)).arcmin

        # de-project
        ypp = yp / np.cos(glx_incl.radian)
        obj_radius = np.sqrt(xp ** 2 + ypp ** 2)  # in arcmin
        obj_dist = Distance(Angle(obj_radius, unit=u.arcmin).radian * glx_dist,
                            unit=glx_dist.unit)

        # Computing PA in disk
        # negative sign needed to get correct orientation from major axis
        obj_phi = Angle(np.arctan2(ypp, -xp), unit=u.rad)
        s = np.where(obj_phi < 0.)[0]
        obj_phi[s] = Angle(2. * np.pi, unit=u.rad) + obj_phi[s]

        return obj_dist, obj_phi
