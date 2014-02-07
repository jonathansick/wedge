#!/usr/bin/env python
# encoding: utf-8
"""
Handles image resampling so that 1) all images share a common pixel grid, and
2) the axis of the wedge is aligned with the positive x-axis of the profile.

This resampling code builds upon that in the Skyoffset and Mo'Astro packages,
and in turn, uses Astromatic's Swarp to carry our the heavy pixel lifting.
"""

from skyoffset.resample import MosaicResampler


def resample_images(image_paths, pa, radec_origin, pixel_scale, work_dir,
        swarp_configs=None,
        weight_paths=None, noise_paths=None, flag_paths=None):
    """Resample a set of images so that the major axis of the profile
    wedge is along the positive x-axis.

    Parameters
    ----------
    image_paths : list
        List of paths (`str`) to FITS images that will be resampled.
    pa : float
        Position angle of the wedge, in degrees.
    radec_origin : tuple
        A tuple of (RA, Dec), giving the location of the wedge origin (i.e.
        the galaxy center).
    pixel_scale : float
        Pixel scale, arcseconds per pixel.
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
