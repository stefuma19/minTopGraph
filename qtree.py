from geom import *


class QTree:
    def __init__(self, dims, maxhsnode, maxdepth):
        self.dims = dims
        self.maxhsnode = maxhsnode
        self.maxdepth = maxdepth
        self.masks = (genmasks(dims))
        self.root = self.createroot()

    def createroot(self):
        root = QNode(None, np.column_stack((np.zeros(self.dims), np.ones(self.dims))))
        self.splitnode(root)

        return root

    def splitnode(self, node):
        mindim = node.mbr[:, 0]
        maxdim = node.mbr[:, 1]

        # The number of quadrants is dependant by the dimensionality
        for quad in range(2 ** self.dims):
            # Convert the quadrant number in binary
            qbin = np.array(list(np.binary_repr(quad, width=self.dims)))

            # Compute new mbr
            child_mindim = np.where(qbin == '0', mindim, (mindim + maxdim) / 2)
            child_maxdim = np.where(qbin == '1', maxdim, (mindim + maxdim) / 2)

            child = QNode(node, np.column_stack((child_mindim, child_maxdim)))

            # Do not build nodes laying above the q1 + q2 + ... + qd = 1 halfspace
            if sum(child_mindim) >= 1:
                child.norm = False

            node.children.append(child)

    def inserthalfspace(self, halfspaces):
        to_search = [self.root]
        self.root.halfspaces = halfspaces

        while len(to_search) > 0:
            current = to_search.pop()

            current.inserthalfspaces(self.masks, current.halfspaces)

            for child in current.children:
                if child.norm:
                    if not child.isleaf() and len(child.halfspaces) > 0:
                        to_search.append(child)
                    elif len(child.halfspaces) > self.maxhsnode and child.depth < self.maxdepth:
                        self.splitnode(child)
                        to_search.append(child)

    def getleaves(self):
        leaves = []
        to_search = [self.root]

        while len(to_search) > 0:
            current = to_search.pop()

            if current.norm:
                if current.isleaf():
                    leaves.append(current)
                else:
                    to_search += current.children

        return leaves


class QNode:
    def __init__(self, parent, mbr):
        self.mbr = mbr
        self.norm = True
        self.order = None
        self.depth = parent.depth + 1 if parent is not None else 0
        self.parent = parent
        self.children = []
        self.covered = []
        self.halfspaces = []

    def isroot(self):
        return self.parent is None

    def isleaf(self):
        return len(self.children) == 0

    def getorder(self):
        self.order = len(self.covered)
        ref = self.parent

        while not ref.isroot():
            self.order += len(ref.covered)
            ref = ref.parent

        return self.order

    def getcovered(self):
        covered = self.covered.copy()
        ref = self.parent

        while not ref.isroot():
            covered += ref.covered
            ref = ref.parent

        return covered

    def inserthalfspaces(self, masks, halfspaces):
        incr = (self.mbr[:, 1] - self.mbr[:, 0]) / 2
        half = (self.mbr[:, 0] + self.mbr[:, 1]) / 2
        pts_mask, nds_mask = masks

        pts = incr * pts_mask + half

        coeff = np.array([hs.coeff for hs in halfspaces])
        known = np.array([hs.known for hs in halfspaces])
        pos = np.where(pts.dot(coeff.T) < known, Position.IN, Position.OUT)

        for hs in range(pos.shape[1]):
            rel = np.where(pos[:, hs] != pos[0, hs])

            cross = np.where(np.sum(nds_mask[rel], axis=0) > 0)
            for c in cross[0]:
                self.children[c].halfspaces.append(halfspaces[hs])

            if pos[0, hs] == Position.IN:
                not_cross = np.where(np.sum(nds_mask[rel], axis=0) == 0)
                for nc in not_cross[0]:
                    self.children[nc].covered.append(halfspaces[hs])
        self.halfspaces = []
