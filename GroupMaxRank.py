import numpy as np
from scipy.spatial import ConvexHull

from maxrank import ba_hd
from qtree import QTree
from geom import *
import queryutils as utils


class Cell:
    def __init__(self, rank, start, record):
        self.rank = rank
        self.start = start
        self.record = record


def g_maxrank_2D(data, group):
    pivot = utils.topk(data.shape[0], data, np.array([0, 1]))

    # Find the worst ranking record among group at q0 = 0
    rank = 0
    highest = None
    for record in group:
        if pivot.index.get_loc(record) > rank:
            rank = pivot.index.get_loc(record) + 1
            highest = record

    # Find its incomparable records
    _, _, incomp = utils.evaluateDominance(data, highest)

    # Compute where it changes position with all of them
    highest_line = Line((data.loc[highest][1] - data.loc[highest][0]), 1, -data.loc[highest][1])
    intersections = []
    for record in incomp:
        line = Line((record[1] - record[0]), 1, -record[1])
        intersections.append((record.name, getLineIntersection(highest_line, line)))

    cells = [Cell(rank, 0, highest)]
    while len(intersections) > 0:
        closest = intersections[0]
        for i in intersections:
            if i[1].coord[0] < closest[1].coord[0]:
                closest = i

        if data.loc[highest][0] >= data.loc[closest[0]][0]:
            rank += 1
        else:
            rank -= 1

        cells.append(Cell(rank, closest[1].coord[0], highest))
        intersections.remove(closest)

        if closest[0] in group:
            highest = closest[0]
            _, _, incomp = utils.evaluateDominance(data, highest)

            # Compute where it changes position with all of them
            highest_line = Line((data.loc[highest][1] - data.loc[highest][0]), 1, -data.loc[highest][1])
            intersections = []
            for record in incomp:
                line = Line((record[1] - record[0]), 1, -record[1])
                pnt = getLineIntersection(highest_line, line)

                if pnt.coord[0] > cells[-1].start:
                    intersections.append((record.name, pnt))

    cells.append(Cell(rank, 1, highest))

    return cells


def g_maxrank_HD(data, group):
    def gen_inv_halfspaces():
        r_d = p.coord[-1]
        r_i = p.coord[:-1]

        hs = []
        for r in worst_pts:
            if r == p:
                continue
            p_d = r.coord[-1]
            p_i = r.coord[:-1]

            hs.append(HalfSpace(r, r_i - r_d - p_i + p_d, p_d - r_d))
        return hs

    def flag_covered_leaves():
        to_search = [qt.root]
        while len(to_search) > 0:
            current = to_search.pop()

            for child in current.children:
                if len(child.covered) > 0:
                    child.norm = False
                elif child.isleaf() and len(child.halfspaces) > 0:
                    child.norm = False

                if child.norm:
                    to_search.append(child)

    data_pts = [Point(record.to_numpy(), _id=record.name) for record in [data.iloc[i] for i in range(data.shape[0])]]

    gen = np.vstack((np.ones(data.shape[1]), data.loc[group].to_numpy()))
    inv_hull = ConvexHull(points=gen, qhull_options='QG0')
    worst_pts = [Point(record.to_numpy(), _id=record.name) for record in [data.loc[group].iloc[i] for i in np.unique(inv_hull.simplices[inv_hull.good]) - 1]]

    maxrank = np.inf
    mincells = []
    for p in worst_pts:
        qt = QTree(data.shape[1] - 1, 0, 7)

        inv_halfspaces = gen_inv_halfspaces()
        qt.inserthalfspace(inv_halfspaces)

        flag_covered_leaves()
        part = qt.getleaves()

        if len(part) > 0:
            minorder, cells = ba_hd(qt, data_pts, p, minorder=maxrank)

            if minorder < maxrank:
                maxrank = minorder
                mincells = cells

            elif minorder == maxrank:
                mincells += cells

    if maxrank < len(group):
        maxrank = len(group)
    return maxrank, mincells
