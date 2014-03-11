#!/usr/bin/env python
# encoding: utf-8
"""
Module for defining a wedge binning.
"""

import math
import numpy as np


def measure_image(image, wedge):
    """Measure an image with the given wedge definition.

    Parameters
    ----------
    image : ndarray
        The image to measure. It should be consistent with the image shape
        expected by the :class:`WedgeBins` definition.
    wedge : :class:`WedgeBins`
        The binning defition.
    
    Returns
    -------
    profile : ndarray
        A `numpy` structured array with the following fields:

        - `area`, the area is square arcseconds
        - `R`, the radial distance at the bin mid-point, in arcseconds
        - `R_inner`, the radial distance at the inside edge, in arcseconds
        - `R_outer`, the radial distance at the outside edge, in arcseconds
        - `median`, the median pixel intensity in the bin
        - `sigma`, the standard deviation of pixel intensities
    """
    dt = [('area', np.float), ('R', np.float), ('R_inner', np.float),
            ('R_outer', np.float), ('median', np.float), ('sigma', np.float)]
    profile = np.zeros(wedge.n_bins, dtype=np.dtype(dt))
    for i, b in enumerate(wedge):
        x1, x2 = b['xlim']
        y1, y2 = b['ylim']
        pixels = image[y1:y2, x1:x2].ravel()
        good = np.where(np.isfinite(pixels))[0]
        pixels = pixels[good]
        median = np.median(pixels)
        sigma = np.std(pixels)
        profile[i]['area'] = b['area']
        profile[i]['R'] = b['r_mid']
        profile[i]['R_inner'] = b['r_inner']
        profile[i]['R_outer'] = b['r_outer']
        profile[i]['median'] = median
        profile[i]['sigma'] = sigma
    return profile


class WedgeBins(object):
    """Defines a wedge binning pattern, assuming the image is aligned with
    the wedge axis.
    
    The bins are square, and defined such that

    .. math::
       W_n = N p (e^{n/N} - 1)

    Parameters
    ----------
    header : :class:`astropy.io.fits.Header`
        The header of a resampled image
    p : int
        Minimum width (in pixels) of the smallest bins.
    N : int
        Width (in pixels) of the largest bin.
    pos_x : bool
        If `True`, then the wedge grows along the positive x-axis. Otherwise,
        the wedge grows leftward.
    """
    def __init__(self, header, p, N, pos_x=True):
        super(WedgeBins, self).__init__()
        self._header = header
        self._p = p
        self._N = N
        self._pos_x = pos_x
        self._bins = self._define_bins()
        self._index = 0

    def __iter__(self):
        return self

    def next(self):
        if self._index == len(self._bins):
            self._index = 0
            raise StopIteration
        self._index += 1
        return self._bins[self._index - 1]

    @property
    def n_bins(self):
        return len(self._bins)

    def _define_bins(self):
        """Create a list of bins, oriented radially."""
        bins = []
        pix_scale = np.abs(self._header['CD1_1']) * 3600.  # arcseconds per px
        nx = self._header['NAXIS1']
        ny = self._header['NAXIS2']
        mid_ix = self._middle_index(nx)
        mid_iy = self._middle_index(ny)
        # Make the central bin, a square around central pixel
        bins.append(self._make_center_bin(mid_ix, mid_iy, pix_scale))
        # Make remainder of bins
        n = 1
        while True:
            if self._pos_x:
                next_bin = self._make_rightward_bin(n, bins[n - 1],
                    mid_ix, mid_iy, pix_scale)
            else:
                next_bin = self._make_leftward_bin(n, bins[n - 1],
                    mid_ix, mid_iy, pix_scale)
            if next_bin is None:
                break
            else:
                bins.append(next_bin)
                n += 1
        return bins

    def _middle_index(self, n):
        """Returns index of pixel in middle, assuming odd integer n.
        1-based for python conventions.
        """
        return int(math.floor(float(n) / 2.))

    def _compute_size(self, n):
        """Compute side-length (pixels) of the logarithmically-growing bin."""
        s = self._N * self._p * (np.exp(float(n) / self._N) - 1.)
        if s < self._p:
            s = self._p
        else:
            s = int(s)
            if not s % 2:  # ensure odd
                s += 1
        return s

    def _compute_ylim(self, s, mid_iy):
        """docstring for _compute_ylim"""
        y1 = mid_iy - int((s - 1) / 2.)
        y2 = mid_iy + int((s - 1) / 2.) + 1
        return [y1, y2]

    def _make_center_bin(self, mid_ix, mid_iy, pix_scale):
        """Define the central bin."""
        # One is added because these are indices.
        x1 = mid_ix - int((self._p - 1) / 2.)
        x2 = mid_ix + int((self._p - 1) / 2.) + 1
        ylim = self._compute_ylim(self._p, mid_iy)
        return self._make_bin_doc([x1, x2], ylim, mid_ix, pix_scale,
                center=True)

    def _make_rightward_bin(self, n, prev, mid_ix, mid_iy, pix_scale):
        """Make the next bin, extending rightward (positive x)."""
        s = self._compute_size(n)
        ylim = self._compute_ylim(s, mid_iy)
        x1 = max(prev['xlim'])
        x2 = x1 + s
        return self._make_bin_doc([x1, x2], ylim, mid_ix, pix_scale)

    def _make_leftward_bin(self, n, prev, mid_ix, mid_iy, pix_scale):
        """Make the next bin, extending leftward (negative x)."""
        s = self._compute_size(n)
        ylim = self._compute_ylim(s, mid_iy)
        x2 = min(prev['xlim'])
        x1 = x2 - s
        return self._make_bin_doc([x1, x2], ylim, mid_ix, pix_scale)
    
    def _make_bin_doc(self, xlim, ylim, mid_ix, pix_scale, center=False):
        """Make a dictionary defining the properties of this bin."""
        # Discard bins entirely outside bounds
        if xlim[1] <= 0:
            return None
        if xlim[0] >= self._header['NAXIS1']:
            return None
        # Make smaller bins if necessary at the end
        if xlim[0] < 0:
            xlim[0] = 0
        if xlim[1] > self._header['NAXIS1']:
            xlim[1] = self._header['NAXIS1']
        if ylim[0] < 0:
            ylim[0] = 0
        if ylim[1] > self._header['NAXIS2']:
            ylim[1] = self._header['NAXIS2']
        A = (xlim[1] - 1 - xlim[0]) * (ylim[1] - 1 - ylim[0]) * pix_scale ** 2.
        r_outer = (xlim[1] - mid_ix - 1.) * pix_scale
        if center:
            r_inner = 0.
        else:
            r_inner = (xlim[0] - mid_ix) * pix_scale
        r_mid = 0.5 * (r_inner + r_outer)
        d = {"xlim": xlim, "ylim": ylim, "area": A,
                "r_outer": r_outer, "r_inner": r_inner, "r_mid": r_mid}
        return d


def plot_bins_in_ax(ax, image, wedge, imshow_args=None, box_args=None):
    """Plot the wedge footprints on an image that has been resampled."""
    import matplotlib as mpl
    ax.imshow(image, **imshow_args)
    verts = []
    for b in wedge:
        xlim = b['xlim']
        ylim = b['ylim']
        poly = [(xlim[0], ylim[0]), (xlim[0], ylim[1]), (xlim[1], ylim[1]),
                (xlim[1], ylim[0])]
        verts.append(poly)
    col = mpl.collections.PolyCollection(verts, **box_args)
    ax.add_collection(col)
