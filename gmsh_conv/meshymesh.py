import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
import pandas as pd

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

def is_on_boundary(x,y):
    """
    Defined on unit square
    """
    lr = (x == 0) + (x == 1)
    ud = (y == 0) + (y == 1)

    return (lr + ud).astype(int)

def find_midpoints(edges):
    mx = np.sum(edges[:,:,0], axis=1)/2
    my = np.sum(edges[:,:,1], axis=1)/2

    return np.array([mx,my]).T

def find_and_replace(A, to_put, to_replace):
    argies = np.zeros(np.shape(A))
    for i in to_replace:
        argies += (A == i)
    A[argies.astype(bool)] = to_put
    return A

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

def create_parallel_mesh(H: float = 0.05, h: int = 4):
    coarse_mesh = tr.triangulate(dict(vertices=[(0,0), (0,1), (1,0), (1,1)]), f"eqa{H}")
    reference_mesh = create_reference_mesh(h)

    btol = []

    n_ref_verts = len(reference_mesh['vertices'])
    n_big_tri = len(coarse_mesh['triangles'])
    n_little_tri = len(reference_mesh['triangles'])

    little_triangles = []
    global_edges = []
    global_triangles = []
    for i, big_triangle in enumerate(coarse_mesh['triangles']):
        arb_triangle, _ = translate(reference_mesh["vertices"], coarse_mesh['vertices'][big_triangle])
        
        little_triangles.append(np.array(arb_triangle))
        plt.scatter(little_triangles[-1][:,0], little_triangles[-1][:,1], marker='o')

        global_edges.append(np.array(reference_mesh['edges']) + i*n_ref_verts)
        global_triangles.append(np.array(reference_mesh['triangles']) + i*n_ref_verts)

        btol.append(list(np.arange(i*n_little_tri, (i+1)*n_little_tri)))

    temp = np.reshape(little_triangles, (-1,2))
    coord_df = pd.DataFrame({'x': np.round(temp[:,0], 5), 'y': np.round(temp[:,1], 5)})
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
        global_edges = find_and_replace(global_edges, k, v)
        global_triangles = find_and_replace(global_triangles, k, v)
        global_vertices = find_and_replace(global_vertices, k, v)

    global_edges = np.sort(global_edges, axis=1)
    dup = pd.DataFrame(global_edges).duplicated(keep='first')
    global_edges = global_edges[~dup]

    global_triangles = np.sort(global_triangles, axis=1)
    dup = pd.DataFrame(global_triangles).duplicated(keep='first')
    global_triangles = global_triangles[~dup]

    dup = pd.Series(global_vertices).duplicated(keep='first')
    coord_df_filtered = coord_df[~dup]
    old = coord_df_filtered.index.values
    coord_df_filtered = coord_df_filtered.reset_index(drop=True)
    new = coord_df_filtered.index.values

    for i,k in enumerate(old):
        global_triangles = find_and_replace(global_triangles, new[i], [k])
        global_edges = find_and_replace(global_edges, new[i], [k])

    a = coord_df_filtered[['x', 'y']].values
    b = a[global_edges]
    midpoints = find_midpoints(b)

    vertex_markers = is_on_boundary(coord_df_filtered.x.values, coord_df_filtered.y.values)
    edge_markers = is_on_boundary(midpoints[:,0], midpoints[:,1])

    mesh = {
        'vertices': np.array([coord_df_filtered.x.values,coord_df_filtered.y.values]).T,
        # 'edges': global_edges,
        'triangles': global_triangles,
        # 'vertex_markers': vertex_markers,
        # 'edge_markers': edge_markers
    
    }

    A = tr.triangulate(mesh, 'er')
    tr.compare(plt, A, mesh)

    A['vertex_markers'] = vertex_markers
    A['edge_markers'] = edge_markers
    A['edges'] = global_edges

    for i in range(len(A["edges"])):
            ii = A["edges"][i][0]
            jj = A["edges"][i][1]

            if ii > jj:
                A["edges"][i][0], A["edges"][i][1] = A["edges"][i][1], A["edges"][i][0]

    # # flipping part + edges
    l = []
    temp = []
    for triangle in A["triangles"]:
        keep, t = get_triangle_edges(triangle, A["edges"])
        l.append(keep)
        temp.append(t)

    keep = np.asarray(l)
    edges_index_inside_triangle = np.asarray(temp)

    A["keep"] = keep
    A["edges_index_inside_triangle"] = edges_index_inside_triangle

    return A, coarse_mesh
