import numpy as np
import scipy.sparse as sp

# Vibe-coded with ChatGPT!


class VoxelSpringSimulator:
    def __init__(
        self,
        coords,
        rest_positions=None,
        init_positions=None,
        stiffness=1000.0,
        mass=1.0,
        damping=0.01,
        gravity=np.array([0.0, 0.0, -9.81]),
        fixed=None,
    ):
        """
        coords          : (N,3) array of integer grid coordinates for adjacency
        rest_positions : optional (N,3) rest‐state positions; defaults to coords
        init_positions : optional (N,3) starting positions; defaults to rest_positions
        stiffness      : spring constant k
        mass           : scalar or length‐N array of per‐voxel masses
        damping        : viscous damping coefficient
        gravity        : 3‐vector
        fixed          : list of vertex‐indices (or boolean mask) to pin
        """
        # store adjacency coords (integers)
        self.coords = np.array(coords, dtype=int)
        self.N = self.coords.shape[0]

        # rest state x0
        if rest_positions is None:
            self.x0 = self.coords.astype(float).copy()
        else:
            rp = np.array(rest_positions, dtype=float)
            if rp.shape != self.coords.shape:
                raise ValueError("rest_positions must match coords shape")
            self.x0 = rp.copy()

        # initial state x
        if init_positions is None:
            self.x = self.x0.copy()
        else:
            ip = np.array(init_positions, dtype=float)
            if ip.shape != self.coords.shape:
                raise ValueError("init_positions must match coords shape")
            self.x = ip.copy()

        # zero initial velocity
        self.v = np.zeros_like(self.x)

        # physical parameters
        self.k = float(stiffness)
        self.damping = float(damping)
        m_arr = mass * np.ones(self.N) if np.isscalar(mass) else np.array(mass, float)
        self.Minv = 1.0 / m_arr
        self.gravity = np.array(gravity, float)

        # build adjacency by Manhattan‐1 neighbors (positive directions)
        coord_to_idx = {tuple(c): i for i, c in enumerate(self.coords)}
        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        pairs = []
        for i, c in enumerate(self.coords):
            for dx, dy, dz in offsets:
                nb = (c[0] + dx, c[1] + dy, c[2] + dz)
                j = coord_to_idx.get(nb)
                if j is not None:
                    pairs.append((i, j))
        self.edges = np.array(pairs, dtype=int)

        # assemble sparse Laplacian L = D - A
        i_idx = np.hstack([self.edges[:, 0], self.edges[:, 1]])
        j_idx = np.hstack([self.edges[:, 1], self.edges[:, 0]])
        data = np.ones_like(i_idx, dtype=float)
        A = sp.coo_matrix((data, (i_idx, j_idx)), shape=(self.N, self.N))
        deg = np.array(A.sum(axis=1)).ravel()
        self.L = sp.diags(deg) - A

        # precompute spring term: c = k * L @ x0
        self.c = self.k * (self.L.dot(self.x0))

        # fixed‐vertex mask
        if fixed is None:
            self.fixed = np.zeros(self.N, dtype=bool)
        else:
            mask = np.zeros(self.N, dtype=bool)
            mask[np.array(fixed, dtype=int)] = True
            self.fixed = mask

    def step(self, dt):
        # spring force
        F_spring = -self.k * (self.L.dot(self.x)) + self.c
        # gravity force
        F_grav = (1.0 / self.Minv)[:, None] * self.gravity
        # damping
        F_damp = -self.damping * self.v

        F_total = F_spring + F_grav + F_damp
        F_total[self.fixed] = 0.0

        # semi‐implicit Euler
        self.v += dt * (F_total * self.Minv[:, None])
        self.v[self.fixed] = 0.0

        self.x += dt * self.v
        self.x[self.fixed] = self.x0[self.fixed]
