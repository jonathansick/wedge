#!/usr/bin/env python
# encoding: utf-8
"""
Handles image resampling so that 1) all images share a common pixel grid, and
2) the axis of the wedge is aligned with the positive x-axis of the profile.

This resampling code builds upon that in the Skyoffset and Mo'Astro packages,
and in turn, uses Astromatic's Swarp to carry our the heavy pixel lifting.
"""

import math

import numpy as np
import astropy.io.fits as fits
# from skyoffset.resample import MosaicResampler


def resample_images(image_paths, radec_origin, pixel_scale, pa,
        wedge_length, wedge_height, work_dir, swarp_configs=None,
        weight_paths=None, noise_paths=None, flag_paths=None):
    """Resample a set of images so that the major axis of the profile
    wedge is along the positive x-axis.

    Parameters
    ----------
    image_paths : list
        List of paths (`str`) to FITS images that will be resampled.
    radec_origin : tuple
        A tuple of (RA, Dec), giving the location of the wedge origin (i.e.
        the galaxy center).
    pixel_scale : float
        Pixel scale, arcseconds per pixel.
    pa : float
        Position angle of the wedge, in degrees.
    wedge_length : float
        Length of the wedge, in arcseconds.
    wedge_height : float
        Maximum height of the wedge, in arcseconds.
    work_dir : str
        Directory where resampled images will be saved.
    swarp_configs : dict
        Optional dictionary of Swarp configurations, to be passed to
        :class:`moastro.astromatic.Swarp`.
    weight_paths : list
        List of paths (`str`) to FITS weight images that will be resampled.
    noise_paths : list
        List of paths (`str`) to FITS noise maps that will be resampled.
    flag_paths : list
        List of paths (`str`) to FITS flag maps that will be resampled.
        Pixels with a value of 1 in the flag maps will be set to NaN.
    """
    if swarp_configs:
        swarp_configs = dict(swarp_configs)
    else:
        swarp_configs = {}

    # Always use a TAN projection
    swarp_configs['PROJECTION_TYPE'] = 'TAN'
    
    # Set the pixel scale
    swarp_configs.update({"PIXELSCALE_TYPE": "MANUAL",
        "PIXEL_SCALE": "{:.2f}".format(pixel_scale)})

    # Set the moasaic center
    # FIXME will actually *not* the center; need to compute offset
    # Or, always make the origin the center so that dual profiles
    # can be build simultaneously to test symmetry?
    swarp_configs['CENTER_TYPE'] = 'MANUAL'
    swarp_configs['CENTER'] = "{:.10f},{:10f}".format(*radec_origin)

    # Set the mosaic image dimensions
    # FIXME
    nx, ny = 1, 1
    swarp_configs['IMAGE SIZE'] = "{:d},{:d}".format(nx, ny)


class TargetWCS(object):
    """A synthesized world coordinate system to resampled images into
    so that the wedge is oriented with the x-axis.

    Parameters
    ----------
    radec_origin : tuple
        A tuple of (RA, Dec), giving the location of the wedge origin (i.e.
        the galaxy center).
    pixel_scale : float
        Pixel scale, arcseconds per pixel.
    pa : float
        Position angle of the wedge, in degrees.
    wedge_box : tuple
        Tuple of maximum wedge (length, width) as a tuple of floats. Units
        are arcseconds.
    """
    def __init__(self, radec_origin, pixel_scale, pa, wedge_box):
        super(TargetWCS, self).__init__()
        self._radec_origin = radec_origin
        self._pixel_scale = float(pixel_scale)
        self._pa_deg = float(pa)
        self._wedge_length = pixel_scale * wedge_box[0]
        self._wedge_height = pixel_scale * wedge_box[1]
        self._wcs_fields = self._compute_wcs()

    def _compute_wcs(self):
        """Compute the target world coordinate system.
        
        Returns
        -------
        wcs_fields : dict
            A dictionary whose keys are FITS header fields, and values are
            the computed WCS.
        """
        # total width, height, in pixels
        NAXIS1 = self._ensure_odd(int(self._wedge_length * 2))
        NAXIS2 = self._ensure_odd(int(self._wedge_height))
        CRPIX1 = self._middle(NAXIS1)
        CRPIX2 = self._middle(NAXIS2)
        CRVAL1 = self._radec_origin[0]  # RA at center
        CRVAL2 = self._radec_origin[1]  # Dec at center

        pa = self._pa_deg * math.pi / 180.
        s = self.pixel_scale / 3600.
        CD1_1 = s * math.cos(pa)
        CD2_1 = -s * math.sin(pa)
        CD1_2 = s * math.sin(pa)
        CD2_2 = s * math.cos(pa)

        return {"NAXIS": 2,
                "RADESYS": 'FK5',
                "EQUINOX": '2000.0',
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "NAXIS1": NAXIS1,
                "NAXIS2": NAXIS2,
                "CRPIX1": CRPIX1,
                "CRPIX2": CRPIX2,
                "CRVAL1": CRVAL1,
                "CRVAL2": CRVAL2,
                "CD1_1": CD1_1,
                "CD2_1": CD2_1,
                "CD1_2": CD1_2,
                "CD2_2": CD2_2}

    def _ensure_odd(self, n):
        """Ensure an integer number is odd."""
        if not n % 2:
            n += 1
        return n

    def _middle(self, n):
        """Given an odd number of pixels, return the pixel in the middle.
        1-based, as is the FITS standard.
        """
        return int(math.ceil(float(n) / 2.))

    def write_fits(self, output_path):
        """Write the WCS to a FITS file, saved to `output_path`.

        The idea is for Swarp to use this FITS file as a target to resample
        other images into.
        
        Parameters
        ----------
        output_path : str
            Path where the blank FITS with WCS will be written.
        """
        n = np.zeros((self._wcs_fields['NAXIS2'], self._wcs_fields['NAXIS1']),
                dtype=np.float)
        hdu = fits.PrimaryHDU(n)
        for k, v in self._wcs_fields.iteritems():
            hdu.header.set(k, v)
        hdu.writeto(output_path, clobber=True)
