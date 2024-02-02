from my_types import *
import triangle as tr
import pandas as pd
from typing import List, Tuple

print("MeshLib imported")
print()


class MeshLib:
    def __init__(self, mesh_dict: dict):
        """
        Initialise mesh object from Triangle dictionary
        """
        for key in mesh_dict:
            setattr(self, key, mesh_dict[key])

        self.h_min, self.h_max, self.h_avg = self.calculate_resolution()

    def convert_to_dict(self):
        """
        Converts attributes to dict
        """
        mesh={}
        mesh['h_min'] = self.h_min
        mesh['h_max'] = self.h_max
        mesh['vertices']=self.vertices
        mesh['triangles']=self.triangles
        mesh['edges']=self.edges
        mesh['vertex_markers']=self.vertex_markers
        mesh['edge_markers']=self.edge_markers
        mesh['keep']=self.keep
        mesh['edges_index_inside_triangle']=self.edges_index_inside_triangle
        mesh['bc_conditions'] = self.bc_conditions
        return mesh

    @classmethod
    def take_parallel_mesh_chain(cls, coarse, fine, bc_conditions):
        coarse = cls.preprocess(coarse, bc_conditions)
        fine = cls.preprocess(fine, bc_conditions)
        return cls(coarse), cls(fine)

    @classmethod
    def preprocess(cls, B, bc_conditions):
        """
        Flips edges, finds the edges in each triangle, then markes the boundary vertices/edges
        """
        # Edge flipping
        for i in range(len(B["edges"])):
            ii = B["edges"][i][0]
            jj = B["edges"][i][1]

            if ii > jj:
                B["edges"][i][0], B["edges"][i][1] = B["edges"][i][1], B["edges"][i][0]

        l = []
        temp = []
        for triangle in B["triangles"]:
            keep, t = cls.get_triangle_edges(triangle, B["edges"])
            l.append(keep)
            temp.append(t)

        keep = np.asarray(l)
        edges_index_inside_triangle = np.asarray(temp)

        # For every triangle you will have 3 numbers that match the corresponding edges,
            # 1 -> You need to flip the edge in the 2D integration
            # 0 -> No flipping is needed
        B["keep"] = keep

        # For every triangle you will have the index of the corresponding edges referring to the attribute edges 
        B["edges_index_inside_triangle"] = edges_index_inside_triangle

        # Marks the vertices and edges
        #  2 -> Neumann
        #  1 -> Dirichlet
        #  0 -> Interior
        mark_vertices,mark_edges = cls.mark_neumann(B, bc_conditions)

        B['vertex_markers'] = mark_vertices
        B['edge_markers'] = mark_edges  
        B['bc_conditions'] = bc_conditions      

        return B

    @classmethod
    def generate_mesh(cls, domain: List[Tuple], refinement_level: float, bc_conditions: str):
        A = dict(vertices=np.array(domain))
        B = tr.triangulate(A, f"q30ea{refinement_level}")

        B["domain"] = domain
        B["refinement_level"] = refinement_level
        B["segments"] = B["edges"]
        B["segment_markers"] = B["edge_markers"]

        B = cls.preprocess(B, bc_conditions)

        return cls(B)
    
    
    @classmethod
    def mark_neumann(cls, mesh, bc_conditions):

        mark_vertices=mesh['vertex_markers'].copy()
        mark_edges=mesh['edge_markers'].copy()

        for v in range(len(mesh['vertex_markers'])):
            mark_vertices[v]=cls.check(mesh['vertices'][v],bc_conditions)

        for v in range(len(mesh['edges'])):

            middle=mesh['vertices'][mesh['edges'][v,1]]+mesh['vertices'][mesh['edges'][v,0]]
            middle=middle/2.0         

            if (0<middle[0]<1.0) and (0<middle[1]<1.0):
                mark_edges[v]=0

            elif (mark_vertices[mesh['edges'][v,0]]==1) and (mark_vertices[mesh['edges'][v,1]]==1):
                    mark_edges[v]=1

            elif (mark_vertices[mesh['edges'][v,0]]==2) and (mark_vertices[mesh['edges'][v,1]]==2):
                mark_edges[v]=2

            elif (mark_vertices[mesh['edges'][v,0]]==1) and (mark_vertices[mesh['edges'][v,1]]==2):
                mark_edges[v]=2

            elif (mark_vertices[mesh['edges'][v,0]]==2) and (mark_vertices[mesh['edges'][v,1]]==1):
                mark_edges[v]=2

            else:
                mark_edges[v]=0
            
        return mark_vertices, mark_edges

    @classmethod
    def check(cls,vertex,bc_conditions):

        #inside we ignore
        if (0.0<vertex[0]<1.0) and (0.0<vertex[1]<1.0):
            return 0


        #if it's inside neumann
        if (bc_conditions[0]=='N'):
            if (vertex[1]==0.0 and 0.0<vertex[0]<1.0):
                return 2
            
        if (bc_conditions[1]=='N'):
            if (vertex[0]==1.0 and 0.0<vertex[1]<1.0):
                return 2
        
        if (bc_conditions[2]=='N'):
            if (vertex[1]==1.0 and 0.0<vertex[0]<1.0):
                return 2
            
        if (bc_conditions[3]=='N'):
            if (vertex[0]==0.0 and 0.0<vertex[1]<1.0):
                return 2

        #corners that are neumann
        if (bc_conditions[0]=='N') and (bc_conditions[1]=='N'):
                if vertex[0]==1.0 and vertex[1]==0.0:
                    return 2

        if (bc_conditions[1]=='N') and (bc_conditions[2]=='N'):
                if vertex[0]==1.0 and vertex[1]==1.0:
                    return 2
                
        if (bc_conditions[2]=='N') and (bc_conditions[3]=='N'):
                if vertex[0]==0.0 and vertex[1]==1.0:
                    return 2
                
        if (bc_conditions[3]=='N') and (bc_conditions[0]=='N'):
            if vertex[0]==0.0 and vertex[1]==0.0:
                return 2

        # if i'm dirichlet boundary case
        return 1 
    
    @classmethod
    def get_triangle_edges(cls, triangle_vertices, edges):
        """
        Helper method for preprocessing
        """
        triangle_edges = []

        keep = np.zeros((3,), dtype=np.int64)

        for i in range(3):
            ii = triangle_vertices[i].copy()
            jj = triangle_vertices[(i + 1) % 3].copy()

            if ii > jj:
                edge = np.array([jj, ii])
            else:
                edge = np.array([ii, jj])

            index = np.where(np.all(edges == edge, axis=1))[0][0]

            triangle_edges.append(index)

            if ii > jj:
                keep[i] = 1

        triangle_edges = np.array(triangle_edges)

        return keep, triangle_edges

    @classmethod
    def refine_mesh(cls, base_mesh: dict, refinement_level: float, bc_conditions, segmentation: bool = False, plot: bool = False):

        assert refinement_level < base_mesh.refinement_level

        A = dict(
            vertices=base_mesh.vertices,
            vertex_markers=base_mesh.vertex_markers,
            triangles=base_mesh.triangles,
        )

        if segmentation:
            A = dict(
                vertices=base_mesh.vertices,
                vertex_markers=base_mesh.vertex_markers,
                triangles=base_mesh.triangles,
                segments=base_mesh.edges,
                segment_markers = base_mesh.edge_markers
            )


        B = tr.triangulate(A, f"eprqa{refinement_level}")
        B["refinement_level"] = refinement_level
        B["domain"] = base_mesh.domain

        B = cls.preprocess(B, bc_conditions)

        return cls(B)

    @classmethod
    def compare(cls, mesh1, mesh2, side_by_side: bool = True):
        """
        Plotting of meshes, either side-by-side or overlap
        """

        if side_by_side:

            plt.figure(figsize=(15,7))

            plt.subplot(121)
            plt.plot(
                mesh1.vertices[:, 0], mesh1.vertices[:, 1], "bo"
            )  
            plt.triplot(
                mesh1.vertices[:, 0],
                mesh1.vertices[:, 1],
                mesh1.triangles,
                color="lightgreen",
            )

            plt.subplot(122)
            plt.plot(
                mesh2.vertices[:, 0], mesh2.vertices[:, 1], "bo"
            )  
            plt.triplot(
                mesh2.vertices[:, 0],
                mesh2.vertices[:, 1],
                mesh2.triangles,
                color="lightgreen",
            )
            plt.axis('off')
            plt.show()

        else:

            plt.figure(figsize=(7,7))

            plt.scatter(
                mesh2.vertices[:, 0], mesh2.vertices[:, 1],
                color='green',
                marker='.',
                zorder=10
            )  
            plt.triplot(
                mesh2.vertices[:, 0],
                mesh2.vertices[:, 1],
                mesh2.triangles,
                color="green",
                linewidth=0.75
            )

            plt.scatter(
                mesh1.vertices[:, 0], mesh1.vertices[:, 1], 
                color='blue',
                marker='.',
                zorder=10
            ) 
            plt.triplot(
                mesh1.vertices[:, 0],
                mesh1.vertices[:, 1],
                mesh1.triangles,
                color="blue",
                linewidth=1
            )
            plt.axis('off')
            plt.show()

    @classmethod
    def generate_regular_mesh_chain(cls, domain: List[Tuple], depth: int, plot: bool = True):
        """
        Generates mesh refinement chain that halfs each triangle per step
        """

        refinement=[0.5/2**(i) for i in range(1,depth)]
        mesh_chain = [cls.generate_mesh(domain, 0.5)]

        if not plot:
            for i in refinement:
                mesh_chain.append(cls.refine_mesh(mesh_chain[-1], i))
            return mesh_chain
        
        mesh_chain[0].plot()
    
        for i in refinement:
            mesh = cls.refine_mesh(mesh_chain[-1], i)
            mesh.plot()
            mesh_chain.append(mesh)
        return mesh_chain  

    def plot(self, triangles: List[int] = False, show: bool = True):
        """
        Plots mesh object of class
        """

        to_plot = self.triangles
        if not triangles is False:
            to_plot = self.triangles[triangles]

        plt.figure(figsize=(4,4))

        plt.scatter(
            self.vertices[:, 0],
            self.vertices[:, 1],
            color='blue',
            marker='.',
            zorder=10
        )  
        plt.triplot(
            self.vertices[:, 0],
            self.vertices[:, 1],
            to_plot,
            color="blue",
            linewidth=0.75
        )
        if show:
            plt.show()

    @classmethod
    def plot_boundary(cls, mesh) -> None:
        """
        Plots the boundary of the mesh, marking the Neumann edges/vertices and the Dirichlet edges/vertices
        """

        if type(mesh) != dict:
            mesh = mesh.convert_to_dict()

        dirichlet=mesh['vertices'][mesh['vertex_markers'][:,0]==1]
        dirichlet_edges=mesh['edges'][mesh['edge_markers'][:,0]==1]

        neumann=mesh['vertices'][mesh['vertex_markers'][:,0]==2]
        neumann_edges=mesh['edges'][mesh['edge_markers'][:,0]==2]
        plt.figure(figsize=(6,6))
        plt.scatter(dirichlet[:,0],dirichlet[:,1])
        plt.scatter(mesh['vertices'][:,0],mesh['vertices'][:,1],marker='*',linewidths=0.01)

        middle_=mesh['vertices'][dirichlet_edges[:,1]]+mesh['vertices'][dirichlet_edges[:,0]]
        middle_=middle_/2
        plt.scatter(middle_[:,0],middle_[:,1])

        plt.scatter(neumann[:,0],neumann[:,1])
        middle=mesh['vertices'][neumann_edges[:,1]]+mesh['vertices'][neumann_edges[:,0]]
        middle=middle/2
        plt.scatter(middle[:,0],middle[:,1])

        # plt.title(f"Boundary marking : {mesh['bc_conditions']}")
        plt.axis('off')
        plt.legend(['Dirichlet vertices ','Vertices','Dirichlet edges','Neumann vertices','Neumann edges'])

    def assign_to_big_triangles(self, coarse_mesh, plot: bool = False):
        """
        Allocates little triangles to big triangles based on centroid of little triangle
        Returns mappings: ltob (little to big) and btol (big to little)
        """

        ltob = np.zeros((len(self.triangles),1)).tolist()

        little_triangle_indices = np.arange(len(self.triangles)).tolist()
        N = len(coarse_mesh.triangles)

        for i in range(N):

            triangle_vertices=coarse_mesh.vertices[coarse_mesh.triangles[i]]
            counter = 0

            for triangle_index in reversed(little_triangle_indices):
                center=np.sum(self.vertices[self.triangles[triangle_index]],axis=0)/3.0
                if self.isInside(center,triangle_vertices)==True:
                    ltob[triangle_index] = i
                    little_triangle_indices.remove(triangle_index)
                    counter=counter+1

        ltob = np.array(ltob)        
        btol = [np.argwhere(ltob == i).flatten() for i in range(N)]
        btol = np.array(btol)

        if plot:
            self.plot_assignment(coarse_mesh, btol)
     
        return ltob, btol

    def plot_assignment(self, coarse_mesh, btol):
        """
        Plots the grouping of little triangles in their big triangles
        """
        plt.figure(figsize=(7,7))
                
        for i in range(len(coarse_mesh.triangles)):
            plt.triplot(self.vertices[:,0], self.vertices[:,1], self.triangles[btol[i]], zorder=-1)
        
        plt.scatter(
            coarse_mesh.vertices[:, 0], coarse_mesh.vertices[:, 1],
            color='black',
            marker='.',
            zorder=10
        )  
        plt.triplot(
            coarse_mesh.vertices[:, 0],
            coarse_mesh.vertices[:, 1],
            coarse_mesh.triangles,
            color="black",
            linewidth=1.5
        )

        plt.show()      

    def calculate_resolution(self):
        """
        Find the min, max, avg resolution of the mesh
        """
        edge_points = self.vertices[self.edges]
        edge_points = np.array(
            [[edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]] for edge in edge_points]
        )
        edge_lengths = np.sqrt((edge_points[:, 0] ** 2) + (edge_points[:, 1] ** 2))
        return np.min(edge_lengths), np.max(edge_lengths), np.mean(edge_lengths)
    
    def summary(self):
        """
        Prints a summary of the mesh and its properties
        """
        print(f"Mesh summary:")
        print(f" - #nodes: {len(self.vertices)}")
        print(f" - #edges: {len(self.edges)}")
        print(f" - #triangles: {len(self.triangles)}")
        print("-------------------------")
        print(f" - h_min: {self.h_min}")
        print(f" - h_max: {self.h_max}")
        print(f" - h_avg: {self.h_avg}")
        print("-------------------------")

    @staticmethod
    def area(x1, y1, x2, y2, x3, y3):
        """
        Helper method to find the area of a triangle
        """
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

    def isInside(self, point, triangle_vertices, eps: float = 1e-8):
        """
        Checks if one triangle is inside another
        """
        x, y = point
        x1, y1 = triangle_vertices[0]
        x2, y2 = triangle_vertices[1]
        x3, y3 = triangle_vertices[2]

        # Calculate area of triangle ABC
        A = self.area(x1, y1, x2, y2, x3, y3)

        # Calculate area of triangle PBC 
        A1 = self.area(x, y, x2, y2, x3, y3)
        
        # Calculate area of triangle PAC 
        A2 = self.area(x1, y1, x, y, x3, y3)
        
        # Calculate area of triangle PAB 
        A3 = self.area(x1, y1, x2, y2, x, y)
        
        # Check if sum of A1, A2 and A3 
        # is same as A
        if np.isclose(A,A1+A2+A3,eps):
            return True 
        else:
            return False

    @staticmethod      
    def is_on_boundary(x,y):
        """
        Defined on unit square
        """
        lr = (x == 0) + (x == 1)
        ud = (y == 0) + (y == 1)

        return (lr + ud).astype(int)

    @staticmethod
    def find_midpoints(edges):
        mx = np.sum(edges[:,:,0], axis=1)/2
        my = np.sum(edges[:,:,1], axis=1)/2

        return np.array([mx,my]).T
    # Generate a mesh on the reference triangle with refinement level N (int)
    @staticmethod
    def create_reference_mesh(N: int):
        N = N+1
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
        return sub_mesh

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
    
    @staticmethod
    def find_and_replace(A, to_put, to_replace):
        argies = np.zeros(np.shape(A))
        for i in to_replace:
            argies += (A == i)
        A[argies.astype(bool)] = to_put
        return A
    
    @classmethod
    def try_and_get_paper_mesh(cls, domain, refinement_level, N, bc_conditions):
        ref_mesh = cls.create_reference_mesh(N)
        coarse_mesh = cls.generate_mesh(domain, refinement_level, bc_conditions)
        btol = []

        n_ref_verts = len(ref_mesh['vertices'])
        n_big_tri = len(coarse_mesh.triangles)
        n_little_tri = len(ref_mesh['triangles'])

        little_triangles = []
        global_edges = []
        global_triangles = []
        for i, big_triangle in enumerate(coarse_mesh.triangles):
            arb_triangle, _ = cls.translate(ref_mesh["vertices"], coarse_mesh.vertices[big_triangle])
            
            little_triangles.append(np.array(arb_triangle))
            # plt.scatter(little_triangles[-1][:,0], little_triangles[-1][:,1], marker='o')

            global_edges.append(np.array(ref_mesh['edges']) + i*n_ref_verts)
            global_triangles.append(np.array(ref_mesh['triangles']) + i*n_ref_verts)

            btol.append(list(np.arange(i*n_little_tri, (i+1)*n_little_tri)))

        fine_mesh = np.reshape(little_triangles, (-1,2))
        coord_df = pd.DataFrame({'x': np.round(fine_mesh[:,0], 5), 'y': np.round(fine_mesh[:,1], 5)})
        coord_df.reset_index(inplace=True, drop=True)

        df1 = (coord_df.groupby(['x', 'y'], sort=False).apply(lambda x: list(x.index))).reset_index(name='idx')
        replace_maps = df1.idx.values
        replace_maps = [r for r in replace_maps if len(r) > 1]
        # to_put = [l[0] for l in replace_maps]
        # to_replace = [l[1:] for l in replace_maps]

        to_put = {l[0]:l[1:] for l in replace_maps}

        to_delete = []
        for l in replace_maps:
            to_delete += l[1:]

        global_edges = np.reshape(global_edges, (-1,2))
        global_triangles = np.reshape(global_triangles, (-1,3))
        global_vertices = coord_df.index.values

        for k, v in to_put.items():
            global_edges = cls.find_and_replace(global_edges, k, v)
            global_triangles = cls.find_and_replace(global_triangles, k, v)
            global_vertices = cls.find_and_replace(global_vertices, k, v)

        global_edges = np.sort(global_edges, axis=1)


        dup = pd.DataFrame(global_edges).duplicated(keep='first')
        print(np.sum(pd.DataFrame(global_edges).duplicated(keep='first')))
        global_edges = global_edges[~dup]
        # print(np.sum(pd.DataFrame(global_edges).duplicated()))

        global_triangles = np.sort(global_triangles, axis=1)
        dup = pd.DataFrame(global_triangles).duplicated(keep='first')
        global_triangles = global_triangles[~dup]

        dup = pd.Series(global_vertices).duplicated(keep='first')
        coord_df_filtered = coord_df[~dup]
        old = coord_df_filtered.index.values
        coord_df_filtered = coord_df_filtered.reset_index(drop=True)
        new = coord_df_filtered.index.values

        for i,k in enumerate(old):
            global_triangles = cls.find_and_replace(global_triangles, new[i], [k])
            global_edges = cls.find_and_replace(global_edges, new[i], [k])

        a = coord_df_filtered[['x', 'y']].values
        b = a[global_edges]
        midpoints = cls.find_midpoints(b)

        vertex_markers = cls.is_on_boundary(coord_df_filtered.x.values, coord_df_filtered.y.values)
        edge_markers = cls.is_on_boundary(midpoints[:,0], midpoints[:,1])

        #TODO: check
        mesh = {
                'vertices': np.array([coord_df_filtered.x.values,coord_df_filtered.y.values]).T,
                'edges': global_edges,
                'triangles': global_triangles,
                'vertex_markers': np.reshape(vertex_markers, (-1,1)),
                'edge_markers': np.reshape(edge_markers, (-1,1))
        }

        A = tr.triangulate(mesh, 'er')
        tr.compare(plt, A, mesh)

        A['vertex_markers'] = np.reshape(vertex_markers, (-1,1))
        A['edge_markers'] = np.reshape(edge_markers, (-1,1))
        A['edges'] = global_edges

        fine_mesh = cls.preprocess(mesh, bc_conditions)
        return coarse_mesh, cls(fine_mesh)

