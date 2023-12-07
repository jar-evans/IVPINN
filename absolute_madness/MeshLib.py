from my_types import *
import triangle as tr

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

    @classmethod
    def generate_mesh(cls, domain: List[Tuple], refinement_level: float):
        A = dict(vertices=np.array(domain))
        B = tr.triangulate(A, f"q30ea{refinement_level}")

        B["domain"] = domain
        B["refinement_level"] = refinement_level
        B["segments"] = B["edges"]
        B["segment_markers"] = B["edge_markers"]

        ### NOTE: ASK M IF THIS IS NEEDED
        ### NOTE: also if this isnt needed then
        ###       can delete get_triangle_edges

        # # edges flipping
        for i in range(len(B["edges"])):
             ii = B["edges"][i][0]
             jj = B["edges"][i][1]

             if ii > jj:
                 B["edges"][i][0], B["edges"][i][1] = B["edges"][i][1], B["edges"][i][0]

        # # flipping part + edges
        l = []
        temp = []
        for triangle in B["triangles"]:
            keep, t = cls.get_triangle_edges(triangle, B["edges"])
            l.append(keep)
            temp.append(t)

        keep = np.asarray(l)
        edges_index_inside_triangle = np.asarray(temp)

        B["keep"] = keep
        B["edges_index_inside_triangle"] = edges_index_inside_triangle

    ### END NOTE

        return cls(B)
    
    @classmethod
    def get_triangle_edges(cls, triangle_vertices, edges):
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
    def refine_mesh(cls, base_mesh: dict, refinement_level: float, segmentation: bool = False, plot: bool = False):

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

        ### NOTE: see above note in generate_mesh
        # # plotting mesh
        # if plot:
        #     tr.compare(plt, A, B)

        # # edges flipping
        for i in range(len(B["edges"])):
            ii = B["edges"][i][0]
            jj = B["edges"][i][1]

            if ii > jj:
                B["edges"][i][0], B["edges"][i][1] = B["edges"][i][1], B["edges"][i][0]

        # flipping part + edges
        l = []
        temp = []
        for triangle in B["triangles"]:
            keep, t = cls.get_triangle_edges(triangle, B["edges"])
            l.append(keep)
            temp.append(t)

        keep = np.asarray(l)
        edges_index_inside_triangle = np.asarray(temp)

        B["keep"] = keep
        B["edges_index_inside_triangle"] = edges_index_inside_triangle


        return cls(B)

    @classmethod
    def compare(cls, mesh1, mesh2, side_by_side: bool = True):

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
            
            plt.show()

    @classmethod
    def generate_regular_mesh_chain(cls, domain: List[Tuple], depth: int, plot: bool = True):

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

    def assign_to_big_triangles(self, coarse_mesh, plot: bool = False):

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
        edge_points = self.vertices[self.edges]
        edge_points = np.array(
            [[edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]] for edge in edge_points]
        )
        edge_lengths = np.sqrt((edge_points[:, 0] ** 2) + (edge_points[:, 1] ** 2))
        return np.min(edge_lengths), np.max(edge_lengths), np.mean(edge_lengths)
    
    def summary(self):
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
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

    def isInside(self, point, triangle_vertices, eps: float = 1e-8):
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