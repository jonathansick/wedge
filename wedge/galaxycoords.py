#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from astropy.coordinates import Distance, Angle
from astropy import units as u


def galaxy_coords(coord, glx_ctr, glx_PA, glx_incl, glx_dist):
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

    return obj_dist, obj_phi, sky_radius.arcsec
