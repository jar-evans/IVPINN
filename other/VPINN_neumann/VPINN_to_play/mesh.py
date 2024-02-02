from my_types import *
import triangle as tr

print('generate mesh lib imported')
print()

def generate_mesh(domain,level_of_refinment,bc):


    A = dict(vertices=np.array(domain))
    B = tr.triangulate(A,'qnea'+str(level_of_refinment))
    
    #plotting mesh
    tr.compare(plt, A, B)
    


    #edges flipping
    for i in range(len(B['edges'])):
        ii=B['edges'][i][0]
        jj=B['edges'][i][1]

        if ii>jj:
                B['edges'][i][0],B['edges'][i][1]=B['edges'][i][1],B['edges'][i][0]
    
    #flipping part + edges
    l=[]
    temp=[]
    for triangle in B['triangles']:
        keep,t=get_triangle_edges(triangle,B['edges'])
        l.append(keep)
        temp.append(t)



    keep=np.asarray(l)
    edges_index_inside_triangle=np.asarray(temp)



    B['keep']=keep
    B['edges_index_inside_triangle']=edges_index_inside_triangle
    B['h']=find_h(B)

    B['domain']=domain

    vertex_markers,edges_markers=mark_neumann(B,bc)

    B['vertex_markers']=vertex_markers
    B['edge_markers']=edges_markers


    return B
    




















def get_triangle_edges(triangle_vertices,edges):
    triangle_edges = []
 

    keep=np.zeros((3,),dtype=np.int64)

    for i in range(3):
        ii=triangle_vertices[i].copy()
        jj=triangle_vertices[(i + 1) % 3].copy()
        

        if(ii>jj):
            edge=np.array([jj,ii])
        else:
            edge=np.array([ii,jj])

        index= np.where(np.all(edges == edge, axis=1))[0][0]
        
        triangle_edges.append(index)

        if ii>jj:
            keep[i]=1

    triangle_edges=np.array(triangle_edges)
        
    return keep,triangle_edges

def find_h(mesh):
    h=-1
    for edges in mesh['edges']:
        vertices=mesh['vertices'][edges]
        h=max(h,np.sqrt((vertices[0,0]-vertices[1,0])**2 +(vertices[0,1]-vertices[1,1])**2))
        return h
    
def find_hs(mesh):
    h_max=-1
    h_min=100
    for edges in mesh['edges']:
        vertices=mesh['vertices'][edges]
        h_max=max(h_max,np.sqrt((vertices[0,0]-vertices[1,0])**2 +(vertices[0,1]-vertices[1,1])**2))
        h_min=min(h_min,np.sqrt((vertices[0,0]-vertices[1,0])**2 +(vertices[0,1]-vertices[1,1])**2))
    return h_max,h_min









def mark_neumann(mesh,bc_conditions):
#1 dirichlet 

    mark_vertices=mesh['vertex_markers'].copy()
    mark_edges=mesh['edge_markers'].copy()


    for v in range(len(mesh['vertex_markers'])):

        mark_vertices[v]=check(mesh['vertices'][v],bc_conditions)



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
        
    return mark_vertices,mark_edges


def check(vertex,bc_conditions):


    #inside i don't care
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