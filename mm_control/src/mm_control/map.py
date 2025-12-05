import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from casadi import MX

from mm_utils import parsing


class SDF2D:
    def __init__(self, config):
        self.dim = 2
        # These params should be the same as in MapInterface
        self.map_coverage = np.array(config["map"]["map_coverage"])
        self.default_val = config["map"]["default_val"]
        self.voxel_size = config["map"]["voxel_size"]
        self.map_size = np.ceil(self.map_coverage / self.voxel_size).astype(int)

        self.xg_sym = MX.sym("xg", self.map_size[0])
        self.yg_sym = MX.sym("yg", self.map_size[1])
        self.v_sym = MX.sym("v", self.map_size[0] * self.map_size[1])
        self.x_sym = MX.sym("x", self.dim)
        interp = cs.interpolant("interpol_linear_x", "linear", self.map_size)
        self.sdf_eqn = interp(
            self.x_sym, cs.vertcat(self.xg_sym, self.yg_sym), self.v_sym
        )
        self.sdf_fcn = cs.Function(
            "map",
            [self.x_sym, self.xg_sym, self.yg_sym, self.v_sym],
            [self.sdf_eqn],
            ["x_query", "x_grid", "y_grid", "value"],
            ["sd(x)"],
        )

        self.hess_eqn, self.grad_eqn = cs.hessian(self.sdf_eqn, self.x_sym)
        grad_eqn_normalized_list = [
            self.grad_eqn,
            self.grad_eqn / cs.norm_2(self.grad_eqn),
        ]
        self.grad_eqn_normalized = cs.conditional(
            cs.norm_2(self.grad_eqn) > 0.01, grad_eqn_normalized_list, 0, False
        )
        self.grad_fcn = cs.Function(
            "map_grad",
            [self.x_sym, self.xg_sym, self.yg_sym, self.v_sym],
            [self.grad_eqn],
        )
        self.grad_fcn_normalzied = cs.Function(
            "map_grad_normalized",
            [self.x_sym, self.xg_sym, self.yg_sym, self.v_sym],
            [self.grad_eqn_normalized],
        )

        self.xg, self.yg = self._get_default_grid()
        self.v = np.ones(self.map_size[0] * self.map_size[1]) * self.default_val

    def update_map(self, xg, yg, v):
        self.xg, self.yg, self.v = xg, yg, v

    def get_params(self):
        return [self.xg, self.yg, self.v]

    def vis(self, x_lim, y_lim, block=True):
        Nx = int(1.0 / 0.1 * (x_lim[1] - x_lim[0])) + 1
        Ny = int(1.0 / 0.1 * (y_lim[1] - y_lim[0])) + 1

        x_1d = np.linspace(x_lim[0], x_lim[1], Nx)
        y_1d = np.linspace(y_lim[0], y_lim[1], Ny)

        X, Y = np.meshgrid(x_1d, y_1d)
        Z = np.zeros(X.shape)
        dZdX = np.zeros((2, X.shape[0], X.shape[1]))
        # This makes the unobserved area free space
        for i in range(len(x_1d)):
            for j in range(len(y_1d)):
                Z[j][i] = self.query_val(x_1d[i], y_1d[j])
                dZdX[:, j, i] = self.query_grad(x_1d[i], y_1d[j]).flatten()

        fig, ax = plt.subplots()
        levels = np.linspace(
            -0.5, self.default_val, int((self.default_val + 0.5) / 0.25) + 1
        )

        cs = ax.contour(X, Y, Z, levels)
        ax.clabel(cs, levels)
        ax.grid()
        ax.quiver(
            X, Y, dZdX[0] / 10, dZdX[1] / 10, scale_units="xy", scale=1, color="gray"
        )

        ax.set_title("Signed Distance Field $sd(x)$")
        ax.set_xlabel("x(m)")
        ax.set_ylabel("y(m)")
        plt.show(block=block)

    def query_val(self, x, y):
        val = self.sdf_fcn(np.vstack((x, y)), self.xg, self.yg, self.v)
        return val

    def query_grad(self, x, y):
        return self.grad_fcn(np.vstack((x, y)), self.xg, self.yg, self.v).toarray()

    def _get_default_grid(self):
        # Limit the map to a certain size around the robot
        xg = np.linspace(
            -self.map_coverage[0] / 2, self.map_coverage[0] / 2, self.map_size[0]
        )
        yg = np.linspace(
            -self.map_coverage[1] / 2, self.map_coverage[1] / 2, self.map_size[1]
        )

        return xg, yg


class SDF3D:
    def __init__(self, config):
        """3D Signed Distance Field Interface for MPC"""

        self.dim = 3
        self.default_val = config["map"]["default_val"]
        self.map_coverage = parsing.parse_array(
            config["map"]["map_coverage"]
        )  # x,y in m
        self.voxel_size = config["map"]["voxel_size"]

        self.map_size = np.ceil(self.map_coverage / self.voxel_size).astype(int)
        self.mul = 10

        self.xg_sym = MX.sym("xg", self.map_size[0])
        self.yg_sym = MX.sym("yg", self.map_size[1])
        self.zg_sym = MX.sym("zg", self.map_size[2])

        self.v_sym = MX.sym("v", self.map_size[0] * self.map_size[1] * self.map_size[2])
        self.x_sym = MX.sym("x", self.dim)
        interp = cs.interpolant("interpol_linear_x", "linear", self.map_size)
        self.sdf_eqn = interp(
            self.x_sym, cs.vertcat(self.xg_sym, self.yg_sym, self.zg_sym), self.v_sym
        )
        self.sdf_fcn = cs.Function(
            "map",
            [self.x_sym, self.xg_sym, self.yg_sym, self.zg_sym, self.v_sym],
            [self.sdf_eqn],
            ["x_query", "x_grid", "y_grid", "z_grid", "value"],
            ["sd(x)"],
        )

        self.hess_eqn, self.grad_eqn = cs.hessian(self.sdf_eqn, self.x_sym)
        self.grad_fcn = cs.Function(
            "map_grad",
            [self.x_sym, self.xg_sym, self.yg_sym, self.zg_sym, self.v_sym],
            [self.grad_eqn],
        )
        self.hess_fcn = cs.Function(
            "map_hess",
            [self.x_sym, self.xg_sym, self.yg_sym, self.zg_sym, self.v_sym],
            [self.hess_eqn],
        )
        self.xg, self.yg, self.zg = self._get_default_grid()
        self.v = (
            np.ones(self.map_size[0] * self.map_size[1] * self.map_size[2])
            * self.default_val
        )

    def update_map(self, xg, yg, zg, v):
        self.xg = xg.copy()
        self.yg = yg.copy()
        self.zg = zg.copy()
        self.v = v.copy()

    def get_params(self):
        return [self.xg.copy(), self.yg.copy(), self.zg.copy(), self.v.copy()]

    def vis(self, x_lim, y_lim, z_lim, block=True):
        Nx = int(1.0 / self.voxel_size * (x_lim[1] - x_lim[0])) + 1
        Ny = int(1.0 / self.voxel_size * (y_lim[1] - y_lim[0])) + 1
        Nz = int(1.0 / self.voxel_size * (z_lim[1] - z_lim[0])) + 1

        dims = ["x", "y", "z"]
        labels = ["x (m)", "y (m)", "z (m)"]
        lims = np.vstack((x_lim, y_lim, z_lim))
        data_1d = []
        shown = []
        not_shown = 0
        for i, N in enumerate([Nx, Ny, Nz]):
            data_1d.append(np.linspace(lims[i][0], lims[i][1], N))
            if N != 1:
                shown.append(i)
            else:
                not_shown = i
        if len(shown) != 2:
            print(f"vis plot 2D data. {len(shown)}D data is given")

        x_idx = shown[0]
        y_idx = shown[1]
        z_idx = not_shown
        X, Y = np.meshgrid(data_1d[x_idx], data_1d[y_idx])
        Z = np.ones_like(X) * data_1d[z_idx][0]

        data_perm = [X, Y, Z]
        data = [0, 0, 0]
        for i, j in enumerate([x_idx, y_idx, z_idx]):
            data[j] = data_perm[i]
        V = self.query_val(*[g.flatten() for g in data])
        V = V.reshape(X.shape)
        G = self.query_grad(*[g.flatten() for g in data])
        G = G.reshape((3, X.shape[0], X.shape[1]))

        fig, ax = plt.subplots()
        levels = np.linspace(-1.0, 2.5, int(3.5 / 0.1) + 1)

        cs = ax.contour(X, Y, V, levels)
        ax.clabel(cs, levels)
        ax.quiver(
            X, Y, G[x_idx] / 10, G[y_idx] / 10, scale_units="xy", scale=1, color="gray"
        )
        ax.grid()
        ax.set_title(
            f"Signed Distance Field sd({dims[x_idx]}, {dims[y_idx]}, {data_1d[not_shown][0]})"
        )
        ax.set_xlabel(labels[x_idx])
        ax.set_ylabel(labels[y_idx])

        plt.show(block=block)

        return ax

    def vis3d(self, x_lim, y_lim, z_lim, block=True):
        Nx = int(1.0 / 0.01 * (x_lim[1] - x_lim[0])) + 1
        Ny = int(1.0 / 0.01 * (y_lim[1] - y_lim[0])) + 1
        Nz = int(1.0 / 0.01 * (z_lim[1] - z_lim[0])) + 1

        lims = np.vstack((x_lim, y_lim, z_lim))
        grid_1d = []
        for i, N in enumerate([Nx, Ny, Nz]):
            grid_1d.append(np.linspace(lims[i][0], lims[i][1], N))

        X, Y, Z = np.meshgrid(*grid_1d)
        V = self.query_val(X.flatten(), Y.flatten(), Z.flatten())
        V = V.reshape(X.shape)
        dVdx = self.query_grad(X.flatten(), Y.flatten(), Z.flatten())
        dVdx = dVdx.reshape((3, X.shape[0], X.shape[1], X.shape[2]))
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection="3d")
        ax1.quiver(
            X,
            Y,
            Z,
            dVdx[0, :, :, :],
            dVdx[1, :, :, :],
            dVdx[2, :, :, :],
            length=0.2,
            arrow_length_ratio=0.2,
            normalize=True,
        )
        print(np.arange(X.flatten().shape[0]).reshape(X.shape))
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_aspect("equal")
        plt.show(block=block)

    def query_val(self, x, y, z):
        input = np.vstack((x, y, z))
        val = self.sdf_fcn(input, self.xg, self.yg, self.zg, self.v).toarray()

        return val

    def query_grad(self, x, y, z):
        return self.grad_fcn(
            np.vstack((x, y, z)), self.xg, self.yg, self.zg, self.v
        ).toarray()

    def query_hessian(self, x, y, z):
        return self.hess_fcn(
            np.vstack((x, y, z)), self.xg, self.yg, self.zg, self.v
        ).toarray()

    def _get_default_grid(self):
        # Limit the map to a certain size around the robot
        max_x = np.around(self.map_coverage[0] / 2, 2)
        min_x = np.around(-self.map_coverage[0] / 2, 2)
        max_y = np.around(self.map_coverage[1] / 2, 2)
        min_y = np.around(-self.map_coverage[1] / 2, 2)
        min_z = 0
        max_z = self.map_coverage[2]

        xg = np.linspace(min_x, max_x, self.map_size[0])
        yg = np.linspace(min_y, max_y, self.map_size[1])
        zg = np.linspace(min_z, max_z, self.map_size[2])

        return xg, yg, zg
