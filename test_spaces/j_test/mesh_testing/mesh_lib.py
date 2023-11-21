import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import pandas as pd

class Mesh:
    def __init__(
        self, domain: list, H: float = None, N: float = None, mesh: dict = None
    ) -> None:
        if domain:
            self.domain = self.create_domain(domain)
            if H:
                self.generate_mesh(H)

            if H and N:
                self.generate_full_mesh(H, N)
        else:
            self.domain = []
            self.mesh = mesh
            self.N = self._get_number_elements()

    def generate_mesh(self, h: float) -> None:
        """Generates Delaunay mesh of given domain."""
        self.mesh = self.create_mesh(self.domain, h)
        self.N = self._get_number_elements()
        self.vertices = self._get_vertices()
        self.edges = self._get_edges()
        self.triangles = self._get_elements()

    @staticmethod
    def create_domain(domain: list) -> dict:
        """Helper method to simplify the definition of a domain. Triangle takes
        domain vertices via a dictionary"""
        return dict(vertices=np.array(domain))

    @staticmethod
    def create_mesh(domain: dict, h: float):
        """Wraps the Triangle methods that performs the triangularisation.

        N.B. e flags the addition of edges to mesh properties, q ensures no element angles
        of sub 20deg, and a permits the inclusion of an area constraint, h. """
        return tr.triangulate(domain, f"enqa{h}")

    def _get_vertices(self):
        """Returns the coordinates of all points in the mesh."""
        return self.mesh["vertices"]

    def _get_edges(self):
        """Returns the node pairs for all edges in the mesh."""
        return self.mesh["edges"]

    def _get_number_elements(self):
        """Returns the number of elements in the mesh (= number of triangles)."""
        return len(self._get_elements())

    def _get_elements(self):
        """Returns the node trios that create the triangles"""
        return self.mesh["triangles"]

    def _get_element_points(self, n: int):
        """
        Returns the vertices of an element of the mesh
        """
        return self._get_vertices()[self._get_elements()[n]]

    def generate_sub_mesh(self, element: int, N: int):
        """Creates"""

        sub_mesh = self.generate_reference_sub_mesh(N)
        sub_mesh["vertices"], J = self.translate(
            sub_mesh["vertices"], self._get_element_points(element)
        )
        return Mesh(False, mesh=sub_mesh)

    def compare(self, other=None):
        if other:
            tr.compare(plt, self.mesh, other.mesh)
        else:
            tr.compare(plt, self.domain, self.mesh)
        plt.show()

    def generate_full_mesh(self, H: float, N: float):
        self.meshed_elements = []
        self.generate_mesh(H)

        for element in range(self.N):
            self.meshed_elements.append(self.generate_sub_mesh(element, N))

    def plot_sub_mesh(self, to_plot: list = [], figsize: tuple = (6, 6)):
        """
        Plots the sub meshes for the indicated elements, can be one element or all
        """

        plt.figure(figsize=figsize)

        plt.plot(
            self._get_vertices()[:, 0], self._get_vertices()[:, 1], "bo"
        )  # Plot element vertices as blue dots

        if not to_plot:
            to_plot = np.arange(0, self.N, 1)

        for element in to_plot:
            plt.triplot(
                (self.meshed_elements[element])._get_vertices()[:, 0],
                (self.meshed_elements[element])._get_vertices()[:, 1],
                (self.meshed_elements[element])._get_elements(),
                color="lightgreen",
            )

        plt.triplot(
            self._get_vertices()[:, 0],
            self._get_vertices()[:, 1],
            self._get_elements(),
        )

        plt.plot(
            self.domain["vertices"][:, 0], self.domain["vertices"][:, 1], "ro"
        )  # Plot original vertices as red dots

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def _get_edge_lengths(self):
        edge_points = self._get_vertices()[self._get_edges()]
        edge_points = np.array(
            [[edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]] for edge in edge_points]
        )
        return np.sqrt((edge_points[:, 0] ** 2) + (edge_points[:, 1] ** 2))

    def max_H(self):
        return max(self._get_edge_lengths())

    def min_H(self):
        return min(self._get_edge_lengths())

    def min_h(self):
        return min(map(lambda sub_mesh: sub_mesh.min_H(), self.meshed_elements))

    def get_refinement_ratio(self):
        return self.min_h() / self.max_H()

    @staticmethod
    def translate(nodes: list, vertices: list):
        """
        Translates given nodes in the reference
        case ([0, 0], [1, 0], [0, 1]) to an arbritrary triangle
        """

        # print(nodes)
        # print(vertices)

        output = np.zeros(shape=np.shape(nodes))

        output[:, 0] = (
            nodes[:, 0] * (vertices[1, 0] - vertices[0, 0])
            + nodes[:, 1] * (vertices[2, 0] - vertices[0, 0])
            + vertices[0, 0]
        )

        output[:, 1] = (
            nodes[:, 0] * (vertices[1, 1] - vertices[0, 1])
            + nodes[:, 1] * (vertices[2, 1] - vertices[0, 1])
            + vertices[0, 1]
        )

        det = (vertices[1, 0] - vertices[0, 0]) * (vertices[2, 1] - vertices[0, 1]) - (
            vertices[2, 0] - vertices[0, 0]
        ) * (vertices[1, 1] - vertices[0, 1])

        return output, det / 2

    # @staticmethod
    # def GLQ():
    #     """
    #     Generate the Gauss-Legendre nodes and weights for
    #     numerical integration
    #     """
    #     return np.array(quad.points), np.array(quad.weights).transpose()

    # # Calculate characteristic size of the mesh (length of longest edge among all triangles)
    # def integrate(self, f) -> float:
    #     I = []
    #     for element in range(self.N):
    #         I.append(self.integrate_element(element, f))
    #     return I, np.sum(I)

    # def integrate_element(self, element, f):
    #     nodes, weights = self.GLQ()
    #     points = self._get_element_points(element)
    #     # print(points)
    #     normalised_points, J = self.translate(nodes, points)
    #     return J * np.sum(f(normalised_points) * weights)

    # def convergence(self, f, exact):
    #     e = []
    #     n = []
    #     for order in range(10, 201, 10):  # Specify the desired order
    #         points, weights = self.GLQ()

    #         n.append(order)
    #         tot = 0.0
    #         _, tot = self.integrate(f)

    #         e.append(np.abs(tot - exact))
    #     e = np.asarray(e)
    #     n = np.asarray(n)

    #     plt.figure(figsize=(8, 6))
    #     plt.loglog(1 / n, e, marker="o", linestyle="-", color="b", label="Error")
    #     plt.loglog(1 / n, 1 / n, marker="o", linestyle="-", color="g", label="1/n")

    #     plt.xlabel("1/n")
    #     plt.title("Convergence Plot")
    #     plt.grid(True, which="both", ls="--")
    #     plt.legend()
    #     plt.show()

    def generate_reference_sub_mesh(self, N: int):
        x = np.linspace(0, 1, N)
        xx, yy = np.meshgrid(x, x)
        xx = xx.flatten()
        yy = yy.flatten()
        filt = xx + yy <= 1
        grid = np.vstack([xx[filt], yy[filt]]).transpose()

        nodes_to_iterate = np.arange(0, (N + 1) * N // 2)
        nodes_to_drop = np.cumsum(np.arange(N, 0, -1)) - 1
        nodes_to_iterate = [i for i in nodes_to_iterate if i not in nodes_to_drop]

        cells = []
        w = N + 1
        for i in nodes_to_iterate:
            if i - 1 in nodes_to_iterate:
                cells.append([i, i + w, i + w - 1])
            else:
                w -= 1
            cells.append([i, i + 1, i + w])

        sub_mesh = tr.triangulate(dict(vertices=grid))
        sub_mesh["triangles"] = cells

        edges = [[[e[0], e[1]], [e[0], e[2]], [e[1], e[2]]] for e in cells]
        # edges = [np.array(e).flatten().tolist() for e in edges]
        edges = np.reshape(np.array(edges).flatten(), (-1, 2)).tolist()
        for e in edges:
            e.sort()
        edge_markers = np.zeros((len(edges), 1))
        dup_free = []
        dup_free_set = set()
        for e in edges:
            if tuple(e) not in dup_free_set:
                dup_free.append(e)
                dup_free_set.add(tuple(e))
            else:
                edge_markers[np.argwhere(dup_free == e)] = 1
        edges = dup_free

        sub_mesh["edges"] = edges

        l = []
        temp = []
        for triangle in cells:
            keep, t = self.get_triangle_edges(triangle, edges)
            l.append(keep)
            temp.append(t)

        l = np.asarray(l)
        temp = np.asarray(temp)

        sub_mesh["flipped"] = l
        sub_mesh["edge_ids"] = temp
        sub_mesh["edge_markers"] = edge_markers
        return sub_mesh


    def get_triangle_edges(self, triangle_vertices, edges):
        triangle_edges = []

        keep = np.zeros((3,), dtype=np.int32)

        for i in range(3):
            ii = triangle_vertices[i]
            jj = triangle_vertices[(i + 1) % 3]

            if ii > jj:
                edge = np.array([jj, ii])
            else:
                edge = np.array([ii, jj])

            index = np.where(np.all(edges == edge, axis=1))[0][0]

            triangle_edges.append(index)

            if ii > jj:
                ii, jj = jj, ii
                keep[i] = 1

        triangle_edges = np.array(triangle_edges)

        return keep, triangle_edges

    def process_dup(self, df):

        a = df.duplicated(subset=['x', 'y'], keep='last')
        b = df.duplicated(subset=['x', 'y'], keep='first')

        A = df.big_triangle_id[a].tolist()
        B = df.big_triangle_id[b].tolist()

        c = [A[i] + B[i] for i in range(len(A))]

        d = df.index[a]

        df['big_triangle_id'][d] = c
        df.drop_duplicates(subset=['x', 'y'], keep='first', inplace=True)
        return df.reset_index(drop=True)


    def find_global_id_edge(self, edge, df):
        res = []
        for coord in edge:
            a = df[['x', 'y']].values
            b = np.argwhere(np.round(coord,6) == a)[:,0]
            u, c = np.unique(b, return_counts=True)
            id = u[c > 1]
            res.append(int(id))

        return res

    def process_dup_edges(self, df):
        
        df['nodes'] = list(map(np.sort, df.nodes.values))
        df['nodes'] = df.nodes.apply(lambda x : tuple(x))
        a = df.nodes.duplicated(keep = 'last')
        b = df.nodes.duplicated(keep = 'first')

        A = df.big_triangle_id[a].tolist()
        B = df.big_triangle_id[b].tolist()

        c = [A[i] + B[i] for i in range(len(A))]

        d = df.index[a]
        df['big_triangle_id'][d] = c

        df = df.drop_duplicates(subset='nodes', keep='first').reset_index(drop=True)
        return df

    def find_global_id_triangle(self, triangle, df):
        res = []
        for coord in triangle:
            a = df[['x', 'y']].values
            b = np.argwhere(np.round(coord,6) == a)[:,0]
            u, c = np.unique(b, return_counts=True)
            id = u[c > 1]
            res.append(int(id))

        return res

    def add_edges(self, nodes, edges_df):
        n = list(nodes)
        n = np.sort(n + n)
        n = n[[0, 3, 1, 4, 2, 5]]
        n = n.reshape((3,2))

        return [edges_df.nodes[edges_df.nodes == tuple(node)].index[0] for node in n]

    def plot_point(self, n: int) -> None:
        plt.figure()

        plt.scatter(self.nodes_df.x.values, self.nodes_df.y.values, marker='+')

        plt.plot(
            self.vertices[:, 0], self.vertices[:, 1], "bo"
        )  # Plot element vertices as blue dots

        to_plot = self.nodes_df.lil_triangle_id[n]
        to_plot = self.triangles_df.nodes[to_plot]

        plt.triplot(
            self._get_vertices()[:, 0],
            self._get_vertices()[:, 1],
            self._get_elements(),
        )

        for lil_triangle in to_plot:
            x = self.nodes_df.x[lil_triangle]
            y = self.nodes_df.y[lil_triangle]
            plt.triplot(x, y, color="green")




        plt.plot(
            self.domain["vertices"][:, 0], self.domain["vertices"][:, 1], "ro"
        )  # Plot original vertices as red dots

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def generate_dfs(self):
        
        nodes_df = pd.DataFrame(columns=['local_id','x', 'y', 'big_triangle_id'])

        for big_triangle_id in range(0, self.N):
            x = np.round(self.meshed_elements[big_triangle_id].mesh['vertices'][:,0], 6)
            y = np.round(self.meshed_elements[big_triangle_id].mesh['vertices'][:,1], 6)

            temp_df = pd.DataFrame({'local_id':np.arange(0,len(x),1),'x':x, 'y':y, 'big_triangle_id':[[big_triangle_id]]*len(x)})
            nodes_df = pd.concat([nodes_df,temp_df])
            nodes_df.reset_index(drop=True, inplace=True)
            nodes_df = self.process_dup(nodes_df)

        boundary = (nodes_df.x == 0) + (nodes_df.x == 1) + (nodes_df.y==0) + (nodes_df.y ==1)
        a = (nodes_df.big_triangle_id.apply(len) > 1).values + boundary

        nodes_df['boundary'] = boundary 
        nodes_df['internal'] = ~a

        edges_df = pd.DataFrame(columns=['local_id','nodes','big_triangle_id'])


        for big_triangle_id in range(0, self.N):

            element = self.meshed_elements[big_triangle_id].mesh

            local_edges = element['edges']
            local_edges = element['vertices'][local_edges]

            local_edges = [self.find_global_id_edge(a, nodes_df) for a in local_edges]

            temp_df = pd.DataFrame({'local_id':np.arange(0,len(local_edges),1),
                                    'nodes':local_edges,
                                    'big_triangle_id':[[big_triangle_id]]*len(local_edges)})

            edges_df = pd.concat([edges_df, temp_df])
            edges_df.reset_index(drop=True, inplace=True)

            edges_df = self.process_dup_edges(edges_df)

        edges_df["nodes"] = edges_df.nodes.apply(list)

        def find_midpoint(x):
            xy = nodes_df.iloc[x][['x','y']]
            return np.round(np.sum(xy.values, axis=0)/2, 6)

        midpoints = edges_df.nodes.apply(find_midpoint)
        edges_df['midpoints'] = midpoints

        triangles_df = pd.DataFrame(columns=['local_id', 'nodes', 'edges', 'big_triangle_id'])

        for big_triangle_id in range(0, self.N):
            element = self.meshed_elements[big_triangle_id].mesh

            local_triangles = element['triangles']
            local_triangles = element['vertices'][local_triangles]

            local_triangles = [self.find_global_id_triangle(a, nodes_df) for a in local_triangles]

            temp_df = pd.DataFrame({'local_id':np.arange(0,len(local_triangles),1),
                                    'nodes':local_triangles,
                                    'big_triangle_id':big_triangle_id})


            triangles_df = pd.concat([triangles_df, temp_df])
            triangles_df.reset_index(drop=True, inplace=True)


        edges_df["nodes"] = edges_df.nodes.apply(tuple)
        edges = [self.add_edges(triangles_df.nodes.values[i], edges_df) for i in triangles_df.index]
        triangles_df['edges'] = edges

        edges_df["nodes"] = edges_df.nodes.apply(list)

        res = []
        for i in nodes_df.index:
            in_res = []
            for j in triangles_df.index:
                if i in triangles_df.nodes.values[j]:
                    in_res.append(j)
            res.append(in_res)

        nodes_df['lil_triangle_id'] = res

        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.triangles_df = triangles_df