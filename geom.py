import numpy as np

from enum import Enum


class Point:
    def __init__(self, coord, _id=None):
        self.id = _id
        self.coord = coord
        self.dims = len(coord)


class HalfSpace:
    def __init__(self, pnt, coeff, known):
        self.pnt = pnt
        self.coeff = coeff
        self.known = known
        self.arr = Arrangement.AUGMENTED
        self.dims = len(coeff)


class HalfLine:
    def __init__(self, pnt):
        self.pnt = pnt
        self.m = pnt.coord[0] - pnt.coord[1]
        self.q = pnt.coord[1]
        self.arr = Arrangement.AUGMENTED
        self.dims = 2

    def get_y(self, x):
        return self.m * x + self.q


class Line:
    def __init__(self, m, boh, q):
        self.boh = boh
        self.m = m
        self.q = q
        # self.arr = Arrangement.AUGMENTED
        # self.dims = 2

    def get_y(self, x):
        return self.m * x + self.q


class Position(Enum):
    """
    Defines the reciprocal position between a point and a halfspace.
    A point can be:
        IN -> Inside the halfspace: satisfies the halfspace disequation
        OUT -> Outside the halfspace: is inside the halfspace complement
        ON -> Lies on the halfspace boundary: satisfies the halfspace equation
    """
    IN = 1
    OUT = -1
    ON = 0


class Arrangement(Enum):
    SINGULAR = 0
    AUGMENTED = 1


# TODO Put this as class method
def genhalfspaces(p, records):
    halfspaces = []
    p_d = p.coord[-1]
    p_i = p.coord[:-1]

    for r in records:
        r_d = r.coord[-1]
        r_i = r.coord[:-1]

        # less-than form
        # s(r) <= s(p)
        halfspaces.append(HalfSpace(r, r_i - r_d - p_i + p_d, p_d - r_d))

    return halfspaces


# TODO Put this as class method
def find_pointhalfspace_position(point, halfspace):
    val = halfspace.coeff.dot(point.coord)

    if val < halfspace.known:
        return Position.IN
    elif val > halfspace.known:
        return Position.OUT
    else:
        return Position.ON


def find_halflines_intersection(r, s):
    if r.m == s.m:
        return None
    else:
        x = (s.q - r.q) / (r.m - s.m)

        return Point([x, r.get_y(x)])


def getLineIntersection(r, s):
    if r.m == s.m:
        return None
    else:
        x = (s.q - r.q) / (r.m - s.m)
        return Point([x, r.get_y(x)])


def genmasks(dims):
    incr = np.full(dims, 0.5)
    pts = np.full((1, dims), 0.5)

    for d in range(dims):
        lower, higher = np.copy(pts), np.copy(pts)
        lower[:, d] -= incr[d]
        higher[:, d] += incr[d]

        pts = np.vstack((pts, lower, higher))
    pts_mask = (pts - incr) / incr

    mbr = np.empty((2 ** dims, dims, 2))
    for quad in range(2 ** dims):
        # Convert the quadrant number in binary
        qbin = np.array(list(np.binary_repr(quad, width=dims)))

        # Compute new mbr
        child_mindim = np.where(qbin == '0', 0, 0.5)
        child_maxdim = np.where(qbin == '1', 1, 0.5)

        mbr[quad] = np.column_stack((child_mindim, child_maxdim))

    nds_mask = np.zeros((pts.shape[0], 2 ** dims), dtype=int)
    for p in range(pts.shape[0]):
        for n in range(2 ** dims):
            if np.all((pts[p] == mbr[n, :, 0]) + (pts[p] == mbr[n, :, 1])):
                nds_mask[p, n] = 1

    return pts_mask, nds_mask
