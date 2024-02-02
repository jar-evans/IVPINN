from my_types import *
import triangle as tr

print("generate mesh lib imported")
print()


def generate_mesh(domain, level_of_refinment):
    A = dict(vertices=np.array(domain))
    B = tr.triangulate(A, "qnea" + str(level_of_refinment))

    # plotting mesh
    tr.compare(plt, A, B)

    # edges flipping
    for i in range(len(B["edges"])):
        ii = B["edges"][i][0]
        jj = B["edges"][i][1]

        if ii > jj:
            B["edges"][i][0], B["edges"][i][1] = B["edges"][i][1], B["edges"][i][0]

    # flipping part + edges
    l = []
    temp = []
    for triangle in B["triangles"]:
        keep, t = get_triangle_edges(triangle, B["edges"])
        l.append(keep)
        temp.append(t)

    keep = np.asarray(l)
    edges_index_inside_triangle = np.asarray(temp)

    B["keep"] = keep
    B["edges_index_inside_triangle"] = edges_index_inside_triangle
    B["h"] = find_h(B)

    B["domain"] = domain

    return B


def get_triangle_edges(triangle_vertices, edges):
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


def find_h(mesh):
    h = -1
    for edges in mesh["edges"]:
        vertices = mesh["vertices"][edges]
        h = max(
            h,
            np.sqrt(
                (vertices[0, 0] - vertices[1, 0]) ** 2
                + (vertices[0, 1] - vertices[1, 1]) ** 2
            ),
        )
        return h


def find_hs(mesh):
    h_max = -1
    h_min = 100
    for edges in mesh["edges"]:
        vertices = mesh["vertices"][edges]
        h_max = max(
            h_max,
            np.sqrt(
                (vertices[0, 0] - vertices[1, 0]) ** 2
                + (vertices[0, 1] - vertices[1, 1]) ** 2
            ),
        )
        h_min = min(
            h_min,
            np.sqrt(
                (vertices[0, 0] - vertices[1, 0]) ** 2
                + (vertices[0, 1] - vertices[1, 1]) ** 2
            ),
        )
    return h_max, h_min
