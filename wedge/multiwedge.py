#!/usr/bin/env python
# encoding: utf-8
"""
Pipeline for dividing an image of an inclined disk galaxy into wedges that
cover the entire galaxy disk.

2014-11-03 - Created by Jonathan Sick
"""

import astropy.io.fits as fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import Distance, Angle, SkyCoord
from astropy import units as u
import numpy as np


class MultiWedge(object):
    """Segment a galaxy image into multiple wedges."""
    def __init__(self, ref_header, segmap=None, pixel_table=None):
        super(MultiWedge, self).__init__()
        self.ref_header = ref_header
        self.ref_wcs = WCS(ref_header)
        self.segmap = segmap
        self.pixel_table = pixel_table

    def segment(self, coord0, d0, incl0, pa0, n_wedges, radial_grid):
        """Segment galaxy image into wedges of `delta` opening angle.

        Radial grid should start > 0. Values between 0. and radial_grid[0]
        are assigned to a central elliptical bin.
        """
        assert radial_grid[0] > 0.
        self._map_pixel_coordinates(coord0, d0, incl0, pa0)
        self._make_segmap(n_wedges, radial_grid)
        self._make_pixel_table(n_wedges, radial_grid)
        print self.pixel_table

    def _map_pixel_coordinates(self, coord0, d0, incl0, pa0):
        shape = (self.ref_header['NAXIS2'], self.ref_header['NAXIS2'])
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
        self.image_r = pixel_R.kpc.reshape(*shape)
        self.image_pa = pixel_PA.deg.reshape(*shape)

        # Make an image of position angle on sky
        self.image_sky_pa = -1. * np.ones(shape, dtype=np.float)
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
        self.image_sky_pa = P.deg.reshape(*shape)

        fits.writeto("_sky_pa_image.fits", self.image_sky_pa, clobber=True)

    def _make_segmap(self, n_wedges, radial_grid):
        pa_delta = 360. / float(n_wedges)
        pa_grid = np.linspace(- pa_delta / 2., 360. - 0.5 * pa_delta,
                              num=n_wedges + 1,
                              endpoint=True)
        pa_grid[0] = 360. - pa_delta / 2.
        pa_segmap = np.zeros(self.image_r.shape, dtype=np.int)
        pa_segmap.fill(-1)
        segmap = np.zeros(self.image_r.shape, dtype=np.int)
        segmap.fill(-1)
        for i in xrange(0, pa_grid.shape[0] - 1):
            print "Segmenting PA {0:d}".format(i)
            if i == 0:
                # special case for first wedge
                inds = np.where(((self.image_sky_pa >= pa_grid[0]) |
                                (self.image_sky_pa < pa_grid[i + 1])) &
                                (self.image_r < radial_grid[-1]))
                pa_segmap[inds] = i
                r_indices = np.digitize(self.image_r[inds], radial_grid,
                                        right=False)
                segmap[inds] = r_indices
            else:
                # for non-wrapping wedges
                inds = np.where((self.image_sky_pa >= pa_grid[i]) &
                                (self.image_sky_pa < pa_grid[i + 1]) &
                                (self.image_r < radial_grid[-1]))
                # because we lose first bin, subtract one off the indices
                r_indices = np.digitize(self.image_r[inds], radial_grid,
                                        right=False) - 1
                pa_segmap[inds] = i
                segmap[inds] = r_indices \
                    + (radial_grid.shape[0] - 1) * (i - 1) \
                    + radial_grid.shape[0]
            # Only do PA+radial binning beyond the first bin
            # r_indices[r_indices == (radial_grid.shape[0] - 1)] = np.nan
            # segmap[inds] = r_indices + radial_grid.shape[0] * i
        # Paint central ellipse
        central_inds = np.where(self.image_r <= radial_grid[0])
        segmap[central_inds] = 0

        self.segmap = segmap
        fits.writeto("_pa_segmap.fits", pa_segmap, clobber=True)
        fits.writeto("_segmap.fits", segmap, clobber=True)

    def _make_pixel_table(self, n_wedges, radial_grid):
        pa_delta = 360. / float(n_wedges)
        n_radii = len(radial_grid) - 1
        n_pixels = n_wedges * n_radii + 1

        # count area of each bin
        s = np.where(self.segmap >= 0)
        pix_count = np.bincount(self.segmap[s].flatten())

        A = self._pixel_scale()
        pix_area = pix_count * A

        # Initialize with central elliptical pixel
        pix_id = [0]
        wedge_id = [0]
        pix_pa = [0.]
        pix_r_inner = [0.]
        pix_r_outer = [radial_grid[0]]
        pix_r_mid = [0.]

        i = 0
        for j in xrange(n_wedges):
            for k in xrange(len(radial_grid) - 1):
                i += 1
                pix_id.append(i)
                wedge_id.append(j)
                pix_pa.append(pa_delta * j)
                pix_r_inner.append(radial_grid[k])
                pix_r_outer.append(radial_grid[k + 1])
                pix_r_mid.append(0.5 * (radial_grid[k + 1] + radial_grid[k]))

        assert len(pix_id) == n_pixels
        assert len(pix_area) == n_pixels

        t = Table((pix_id, wedge_id, pix_pa, pix_r_mid,
                   pix_r_inner, pix_r_outer, pix_area),
                  names=('ID', 'W_ID', 'PA', "R_kpc", 'R_inner', 'R_outer',
                         'area'))
        self.pixel_table = t

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

    def _pixel_scale(self):
        if 'CDELT' in self.ref_header:
            pix_scale = self.ref_header['CDELT'] * 3600.
        else:
            pix_scale = np.sqrt(self.ref_header['CD1_1'] ** 2.
                                + self.ref_header['CD1_2'] ** 2.) * 3600.
        return pix_scale
