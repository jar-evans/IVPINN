from my_types import *
from MeshLib import *
from interpolator import *
import time
from quad import *


class IVPINN():

    def __init__(self, pb, params, mesh, verbose, coarse_mesh, NN = None):

        super().__init__()

        # accept problem parameters
        self.pb = pb
        self.params = params
        self.n_test = params['N_test']
        self.r_interpolation=params['r_interpolation']
        self.verbose=verbose 

        # convert meshes to dicts
        self.mesh = mesh.convert_to_dict()
        self.coarse_mesh = coarse_mesh.convert_to_dict()

        self.boundary_function = self.create_boundary_function()


        # collect mapping from big triangles to little triangles
        _,self.btol=mesh.assign_to_big_triangles(coarse_mesh, plot=True)

        # find number of points, edges, triangles
        self.n_vertices=len(self.mesh['vertices'])
        self.n_edges=len(self.mesh['edges'])
        self.n_triangles=len(self.mesh['triangles'])
        self.n_big_triangles=len(self.coarse_mesh['triangles'])

        # calculating the number of DoF for first order and second order cases
        if self.n_test>=2:
            self.dof=(len(self.mesh['vertex_markers'][self.mesh['vertex_markers']!=1.0]))+(len(self.mesh['edge_markers'][self.mesh['edge_markers']!=1.0]))
        else:
            self.dof=(len(self.mesh['vertex_markers'][self.mesh['vertex_markers']!=1.0]))

        # precomputes the RHS
        self.pre_compute()

        # add the neural network to the class if given at initialisation
        if NN:
            self.set_NN(NN)

        # initialising the degree of freedom vectors for vertices
        self.sum_of_vectors_vertices = tf.Variable(tf.zeros((self.n_vertices,1),dtype=tf_type)) 
        self.sum_of_vectors_vertices.assign(tf.zeros_like(self.sum_of_vectors_vertices))   

        # initialising the degree of freedom vectors for edges (for second order)
        if self.n_test>=2:
            self.sum_of_vectors_edges = tf.Variable(tf.zeros((self.n_edges,1),dtype=tf_type))       
            self.sum_of_vectors_edges.assign(tf.zeros_like(self.sum_of_vectors_edges))

        # prints summary of mesh and problem parameters
        self.summary()

    def set_NN(self, NN, LR=0.001):

        self.NN = NN

        # take trainable vars
        self.vars = self.NN.trainable_variables

        # set optimiser
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)


    @tf.function  # unnecessary
    def eval_NN(self, x):
        """input tensor of size (n,2) returns tensor of size (n,1)"""
        return self.NN(x)
    
    
    def u_NN_BC(self, x, y):
        """
        Evaluates NN with BC imposed in numpy
        """
        eval=tf.constant([[x,y]],dtype=tf.float64)
        return self.NN_imposeBC(eval).numpy()[0,0] 
    
    def create_boundary_function(self): 
        """
        Supresses solution on Dirichlet boundary
        """

        functions = []

        boundary_markings = self.mesh['bc_conditions']     

        if boundary_markings[0] == 'D':
            functions.append(self.boundary_function_bottom)
        else:
            function.append(1)

        if boundary_markings[1] == 'D':
            functions.append(self.boundary_function_right)
        else:
            function.append(1)

        if boundary_markings[2] == 'D':
            functions.append(self.boundary_function_top)
        else:
            function.append(1)

        if boundary_markings[3] == 'D':
            functions.append(self.boundary_function_left)
        else:
            function.append(1)

        def h(*args):
            return function[0](*args)*function[1](*args)*function[2](*args)*function[3](*args)
        
        return h

        # return tf.expand_dims(x[:,0]*(1-x[:,0]),axis=1)
    
    @staticmethod
    def boundary_function_bottom(x):
        return x[:,1]
    
    @staticmethod
    def boundary_function_top(x):
        return 1-x[:,1]
    
    @staticmethod
    def boundary_function_left(x):
        return x[:,0]
    
    @staticmethod
    def boundary_function_right(x):
        return 1-x[:,0]


    def NN_imposeBC(self,x):
        eval = self.NN(x)
        boundary = tf.expand_dims(self.boundary_function(x), axis=1)
        return eval*boundary + self.bc_model(x)
    

    # @tf.function
    def eval_grad_NN_BC(self,x):
        """
        input tensor of size (n,2)
        returns tensor of size (n,2) -> on each row you have first der x and then der y
        """

        with tf.GradientTape() as tape:
            tape.watch(x)
            res=self.NN_imposeBC(x)
        grad=tape.gradient(res,x)
        return grad
    
    def set_bc_model(self, bc_model):
        self.bc_model = bc_model

    def standard_bc(self,x):
        # return tf.expand_dims((tf.sin(x[:,0])*tf.sin(1-x[:,0])*tf.sin(x[:,1])*tf.sin(1-x[:,1])+1.0)*self.pb.u_exact(x[:,0],x[:,1]),axis=1)
        return tf.expand_dims(self.pb.dirichlet(x[:,0],x[:,1]), axis=1)
    

    def helper(self):
        """
        Precomputes the points to evaluate for each coarse triangle
        """

        n_big_triangles=self.n_big_triangles
        r=self.n_test

        x_to_eval_global = []

        for big_traingle in range(n_big_triangles):

            vertices_big=self.coarse_mesh['vertices'][self.coarse_mesh['triangles'][big_traingle]]

            B_i, c_i, det_i, B_D_i, B_DD_i, B_inv_i=self.B.change_of_coordinates(vertices_big)

            x_to_eval=B_i@self.B.Nodes.T +c_i

            x_to_eval=np.transpose(x_to_eval)

            x_to_eval_global.append(x_to_eval)

        self.x_to_eval_global = tf.constant(x_to_eval_global, dtype=tf_type)


    def custom_loss_IVPINN(self):
        """
        Custom loss loop for the IVPINN case
        """

        n_big_triangles=self.n_big_triangles

        # create global evaluations
        x_to_eval_global = tf.reshape(self.x_to_eval_global, (-1,2))
        eval_global = self.NN_imposeBC(x_to_eval_global)
        eval_global = tf.reshape(eval_global, (n_big_triangles, -1, 1))


        # finding quadrature for all small triangles        
        xy_quad_total =self.xy_quad_total.numpy()
        w_quad = tf.concat([self.w_quad.T,self.w_quad.T], axis=0)

        # create helper variables for boundary integration
        w_ = tf.reshape(self.w_quad, (-1,))
        test = self.test
        temp = tf.reshape(self.test, (tf.shape(self.test)[0], 1, tf.shape(self.test)[1]))
        test_double = tf.concat([temp, temp], axis=1)

        # take degree
        r=self.n_test

        # init rhs and dof vector
        F_total_vertices=self.F_total_vertices
        sum_of_vectors_vertices=self.sum_of_vectors_vertices

        # init rhs and dof vector for edges in the 2nd order case
        if r > 1:
            F_total_edges=self.F_total_edges   
            sum_of_vectors_edges=self.sum_of_vectors_edges

        # taking gradient of test functions
        grad_test=self.grad_test 

        # for each big triangle
        for big_traingle in range(n_big_triangles):

            # get the vertices
            vertices_big=self.coarse_mesh['vertices'][self.coarse_mesh['triangles'][big_traingle]]
            
            # find the change of coord matrices for THIS big triangle
            B_i, c_i, det_i, B_D_i, B_DD_i, B_inv_i=self.B.change_of_coordinates(vertices_big)

            # extracting big triangle evaluations of the network
            eval = eval_global[big_traingle]

            # finding all the small triangles in THIS big triangle
            little_triangles=self.btol[big_traingle]

            # and then for all the little triangles
            for index,triangle in enumerate(self.mesh['triangles'][little_triangles]):

                # get the change of coord matrices
                B,c,J,B_D,B_DD,B_inv=self.b.change_of_coordinates(self.mesh['vertices'][triangle])

                # transforming gradient of test functions from ref. to our little triangle
                grad_test_elem=B_D @ grad_test

                # x = b*x_ref + c
                x_ref = B_inv_i @ (np.transpose(xy_quad_total[little_triangles[index]]) - c_i)

                x_ref = np.transpose(x_ref)

                # finding the gradient of the NN on our quadrature points on the reference triangle
                grad_elem_x=self.B.interpolate_dx_tf(x_ref,eval)  
                grad_elem_y=self.B.interpolate_dy_tf(x_ref,eval)  

                # allocate the dx, dy to one array of (2, n_quad)
                grad_elem=tf.concat([grad_elem_x,grad_elem_y],axis=1)
                grad_elem=tf.transpose(grad_elem)

                # transform the gradient on the reference to the big triangle
                grad_elem = B_D_i @ grad_elem   

                # interpolates the evaluations
                elem=self.B.interpolate_tf(x_ref,eval) 
                elem = tf.reshape(elem, (-1,)) 
                              
                # fetch the problem coefficeinets from the precomputed global evaluations
                mu_elem = self.mu[little_triangles[index]]
                beta_elem = self.beta[little_triangles[index]]
                sigma_elem = self.sigma[little_triangles[index]]

                # then we take the integral        
                v0= J*tf.reduce_sum(w_quad*grad_test_elem[0]*grad_elem*mu_elem) + J*tf.reduce_sum(w_quad*beta_elem*grad_elem*test_double[0]) + J*tf.reduce_sum(w_*sigma_elem*test[0]*elem)
                v1= J*tf.reduce_sum(w_quad*grad_test_elem[1]*grad_elem*mu_elem) + J*tf.reduce_sum(w_quad*beta_elem*grad_elem*test_double[1]) + J*tf.reduce_sum(w_*sigma_elem*test[1]*elem)
                v2= J*tf.reduce_sum(w_quad*grad_test_elem[2]*grad_elem*mu_elem) + J*tf.reduce_sum(w_quad*beta_elem*grad_elem*test_double[2]) + J*tf.reduce_sum(w_*sigma_elem*test[2]*elem)

                # supress result if node is dirichlet
                if (self.mesh['vertex_markers'][triangle[0]]==1):
                    v0=v0*0.0
                if (self.mesh['vertex_markers'][triangle[1]]==1):
                    v1=v1*0.0
                if (self.mesh['vertex_markers'][triangle[2]]==1):
                    v2=v2*0.0

                # allocated results to correct position in dof vector
                indices_vertices = tf.constant([[triangle[0]], [triangle[1]], [triangle[2]]])
                updates_vertices = [v0, v1, v2]

                scatter_vertices = tf.scatter_nd(indices_vertices, updates_vertices, [self.n_vertices])
                scatter_vertices_=tf.expand_dims(scatter_vertices,axis=1)
                sum_of_vectors_vertices = tf.reduce_sum([sum_of_vectors_vertices, scatter_vertices_], axis=0)


                # repeat process for the bases on the edges in the 2nd order case
                if r > 1:

                    l0= J*tf.reduce_sum(w_quad*grad_test_elem[3]*grad_elem*mu_elem) + J*tf.reduce_sum(w_quad*beta_elem*grad_elem*test_double[3]) + J*tf.reduce_sum(w_*sigma_elem*test[3]*elem)
                    l1= J*tf.reduce_sum(w_quad*grad_test_elem[4]*grad_elem*mu_elem) + J*tf.reduce_sum(w_quad*beta_elem*grad_elem*test_double[4]) + J*tf.reduce_sum(w_*sigma_elem*test[4]*elem)
                    l2= J*tf.reduce_sum(w_quad*grad_test_elem[5]*grad_elem*mu_elem) + J*tf.reduce_sum(w_quad*beta_elem*grad_elem*test_double[5]) + J*tf.reduce_sum(w_*sigma_elem*test[5]*elem)

                    if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][little_triangles[index]][0]]==1):
                        l0=l0*0.0

                    if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][little_triangles[index]][1]]==1):
                        l1=l1*0.0

                    if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][little_triangles[index]][2]]==1):
                        l2=l2*0.0
                    
                    indices_edges = tf.constant([[self.mesh['edges_index_inside_triangle'][little_triangles[index]][0]], [self.mesh['edges_index_inside_triangle'][little_triangles[index]][1]], [self.mesh['edges_index_inside_triangle'][little_triangles[index]][2]]])
                    updates_edges=[l0,l1,l2]

                    scatter_edges = tf.scatter_nd(indices_edges, updates_edges, [self.n_edges])
                    scatter_edges_=tf.expand_dims(scatter_edges,axis=1)    
                    sum_of_vectors_edges = tf.reduce_sum([sum_of_vectors_edges, scatter_edges_], axis=0)


        # return the dof in either 1st or 2nd order cases
        if r > 1:
                return (tf.reduce_sum(tf.square(sum_of_vectors_vertices-F_total_vertices))+tf.reduce_sum(tf.square(sum_of_vectors_edges-F_total_edges)))/self.dof
        else:
                return (tf.reduce_sum(tf.square(sum_of_vectors_vertices-F_total_vertices)))/self.dof

    
    def loss_gradient(self):       
        with tf.GradientTape() as tape:
            loss = self.custom_loss_IVPINN()
        gradient = tape.gradient(loss, self.NN.trainable_variables)
        return loss, gradient

    @tf.function
    def gradient_descent(self):
        loss, gradient = self.loss_gradient()
        self.optimizer.apply_gradients(zip(gradient, self.NN.trainable_variables))
        return loss

    def train(self, iter, LR, bc_model=None):

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

        self.helper()

        if bc_model==None:
            self.bc_model=self.standard_bc
        else:
            self.bc_model=bc_model


        history = []

        start_time = time.time()
        for i in range(iter+1):
            self.sum_of_vectors_vertices.assign(tf.zeros_like(self.sum_of_vectors_vertices))

            if self.n_test > 1:       
                self.sum_of_vectors_edges.assign(tf.zeros_like(self.sum_of_vectors_edges))             

            loss = self.gradient_descent()  #add other losses 
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f'Iteration: {i}', f'loss: {loss.numpy():0.10f}', f'time: {elapsed}')

                history.append(loss)
                start_time = time.time()
                
        return history


    def residual_summary(self,residual_vertices, residual__edges = None):

        rv=residual_vertices[self.F_total_vertices!=0.0]        

        print('residual of vertices : ')
        print('-->max  = ',np.max(rv))
        print('-->min  = ',np.min(rv))
        print('-->mean = ',np.mean(rv))

        if self.n_test > 1:
            re=residual__edges[self.F_total_edges!=0.0]

            print('residual of edges : ')
            print('-->max  = ',np.max(re))
            print('-->min  = ',np.min(re))
            print('-->mean = ',np.mean(re))

    def pre_compute(self):

        self.generate_quadrature_points()

        self.evaluate_test_and_inter_functions()

        self.construct_RHS()

    def evaluate_test_and_inter_functions(self):
    
        self.b=interpolator(self.params['N_test'],self.verbose,True,points=self.points)
        self.B=interpolator(self.r_interpolation,False,False,points=None)

        grad=[]
        test=[]

        for i in range(self.b.n):
            elem=np.stack([self.b.Base_dx[:,i],self.b.Base_dy[:,i]])
            grad.append(elem)
            test.append(self.b.Base[:,i])


        self.grad_test=tf.constant(grad, dtype=tf_type)
        self.test=tf.constant(test, dtype=tf_type)

        print(tf.shape(self.test))

    def generate_quadrature_points(self):
        """
        here you will have col vectors with correct stuff defined on the ref triangle
        self.x_quad
        self.y_quad 
        self.w_quad
        """
        self.xy_quad =np.array(points,dtype=np.float64)
        self.w_quad = np.array(weights,dtype=np.float64)
        self.x_quad = np.reshape(self.xy_quad[:,0], (len(self.xy_quad), 1))
        self.y_quad = np.reshape(self.xy_quad[:,1], (len(self.xy_quad), 1))

        self.points=self.xy_quad
        
        self.w_quad = np.reshape(self.w_quad, (len(self.w_quad), 1))

    def line_transformation(self, vert0, vert1, ref):
        temp1 = (vert1[0] - vert0[0])*ref + vert0[0]
        temp2 = (vert1[1] - vert0[1])*ref + vert0[1]
        temp1 = np.reshape(temp1, (-1,1))
        temp2 = np.reshape(temp2, (-1,1))
        temp3 = np.concatenate([temp1, temp2], axis=1)

        return temp3

    def construct_RHS(self):
        """
        Pre-compute the RHS of the formulation
        """

        F_total_vertices = np.zeros((self.n_vertices,1),dtype=np.float64)
        F_total_edges = np.zeros((self.n_edges,1),dtype=np.float64)
        r=self.b.r

        xy_quad_total = []
        J_total = []
        # F_total=[]
        
        neumann_points_ref=self.b.neumann_nodes
        neumann_weights=self.b.neumannn_weights

        mu = []
        sigma = []
        beta = []

        points_edge1_ref=np.hstack([neumann_points_ref,np.zeros_like(neumann_points_ref)])#,dtype=np_type)
        points_edge2_ref=np.hstack([1.0-neumann_points_ref,neumann_points_ref])#,dtype=np_type)
        points_edge3_ref=np.hstack([np.zeros_like(neumann_points_ref),neumann_points_ref[::-1]])#,dtype=np_type)
 

        for index,triangle in enumerate(self.mesh['triangles']):
            F_element=[]
            # x_element=[]
            J_element=[]
            x_quad=self.points

            vertices=self.mesh['vertices'][triangle] #3x2 

            # get quadrature points in arb. element and get jacobian
            B,c,J,B_D,B_DD,B_inv=self.b.change_of_coordinates(self.mesh['vertices'][triangle])

            xy_quad_element=(B@x_quad.T +c).T
      
            J_element.append(J)

            # evaluate f on arb. quad points
            f_quad_element = self.pb.f(xy_quad_element[:, 0], xy_quad_element[:, 1])
            mu_element = self.pb.mu(xy_quad_element[:, 0], xy_quad_element[:, 1])
            b1, b2 = self.pb.beta(xy_quad_element[:, 0], xy_quad_element[:, 1])
            sigma_element = self.pb.sigma(xy_quad_element[:, 0], xy_quad_element[:, 1])

            beta_element = [b1, b2]

            mu.append(mu_element)
            sigma.append(sigma_element)
            beta.append(beta_element)

            # do the integral and appnd to total list
            #print([J*np.sum(self.w_quad[:,0]*self.b.Base[:,r]*f_quad_element) for r in range(self.b.n)])
            F_element=[J*np.sum(self.w_quad[:,0]*self.b.Base[:,r]*f_quad_element) for r in range(self.b.n)]

            F_element=np.array(F_element,dtype=np_type)
            J_element=np.array(J_element,dtype=np_type)

            if (self.mesh['vertex_markers'][triangle[0]]!=1):
                    F_total_vertices[triangle[0]]+=F_element[0]


            if (self.mesh['vertex_markers'][triangle[1]]!=1):
                    F_total_vertices[triangle[1]]+=F_element[1]
 

            if (self.mesh['vertex_markers'][triangle[2]]!=1):
                    F_total_vertices[triangle[2]]+=F_element[2]

                    
            # for the first edge int he triangle -> is it neumann?
            if (self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]==2):

                # calc det by eucl norm
                det=np.sqrt(np.square(vertices[0,0]-vertices[1,0]) +np.square(vertices[0,1]-vertices[1,1]))
            
                # is the first node on the edge neumann?
                if (self.mesh['vertex_markers'][triangle[0]]==2):
                    
                    # identify test function in the basis
                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[0]=1.0

                    #translate reference points to current triangle
                    points_edge1=(B@points_edge1_ref.T +c).T
                    # evaluate neumann function on edge
                    res=self.pb.neumann(points_edge1[:,0],points_edge1[:,1])
                    res=np.reshape(res,(-1,))

                    jacob = (B_inv@(points_edge1.T - c)).T

                    # evaluate test function on edge
                    test=self.b.interpolate(jacob,select)

                    test = tf.reshape(test, (-1,))
                    neumann_weights = tf.reshape(neumann_weights, (-1,))

                    # integrate the test function against the neumann function on the edge
                    F_total_vertices[triangle[0]]+=det*tf.reduce_sum(neumann_weights*res*test)

                    print(f"Triangle #{index} 1st edge, vertex 1")

                if self.mesh['vertex_markers'][triangle[1]]==2:

                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[1]=1.0
                    points_edge1=(B@points_edge1_ref.T +c).T

                    res=self.pb.neumann(points_edge1[:,0],points_edge1[:,1])
                    res=np.reshape(res,(-1,))
                    jacob = (B_inv@(points_edge1.T - c)).T

                    # evaluate test function on edge
                    test=self.b.interpolate(jacob,select)      

                    test = tf.reshape(test, (-1,))
                    neumann_weights = tf.reshape(neumann_weights, (-1,))           
                   
                    F_total_vertices[triangle[1]]+=det*tf.reduce_sum(neumann_weights*res*test)

                    print(f"Triangle #{index} 1st edge, vertex 2")

            if (self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]==2):

                det= np.sqrt(np.square(vertices[1,0]-vertices[2,0]) +np.square(vertices[1,1]-vertices[2,1]))

                if (self.mesh['vertex_markers'][triangle[1]]==2):
                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[1]=1.0
                    # det= np.sqrt(np.square(vertices[1,0]-vertices[2,0]) +np.square(vertices[1,1]-vertices[2,1]))
                    points_edge2=(B@points_edge2_ref.T +c).T#

                    res=self.pb.neumann(points_edge2[:,0],points_edge2[:,1])
                    res=np.reshape(res,(-1,1))
                    test=self.b.interpolate(points_edge2_ref,select)
                    
                    F_total_vertices[triangle[1]]+=det*np.sum(neumann_weights*res*test)

                    print('2nd edge, vertex 2')

                if (self.mesh['vertex_markers'][triangle[2]]==2):
                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[2]=1.0
                    
                    points_edge2=(B@points_edge2_ref.T +c).T#

                    res=self.pb.neumann(points_edge2[:,0],points_edge2[:,1])
                    res=np.reshape(res,(-1,1))
                    test=self.b.interpolate(points_edge2_ref,select)
                    
                    F_total_vertices[triangle[2]]+=det*np.sum(neumann_weights*res*test)

                    print('2nd edge, vertex 3')
    
            if (self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]==2):

                det= np.sqrt(np.square(vertices[0,0]-vertices[2,0]) +np.square(vertices[0,1]-vertices[2,1]))

                if (self.mesh['vertex_markers'][triangle[2]]==2):
                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[2]=1.0
                    
                    points_edge3=(B@points_edge3_ref.T +c).T#

                    res=self.pb.neumann(points_edge3[:,0],points_edge3[:,1])
                    res=np.reshape(res,(-1,1))
                    test=self.b.interpolate(points_edge3_ref,select)

                    F_total_vertices[triangle[2]]+=det*np.sum(neumann_weights*res*test)

                    print('3rd edge, vertex 3')

                if (self.mesh['vertex_markers'][triangle[0]]==2):
                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[0]=1.0
                    
                    points_edge3=(B@points_edge3_ref.T +c).T#

                    res=self.pb.neumann(points_edge3[:,0],points_edge3[:,1])
                    res=np.reshape(res,(-1,1))
                    test=self.b.interpolate(points_edge3_ref,select)
                    
                    F_total_vertices[triangle[0]]+=det*np.sum(neumann_weights*res*test)

                    print('3rd edge, vertex 1')


            if(r>=2):
                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]!=1):
                    F_total_edges[self.mesh['edges_index_inside_triangle'][index][0]]+=F_element[3]

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]!=1):
                    F_total_edges[self.mesh['edges_index_inside_triangle'][index][1]]+=F_element[4]

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]!=1):
                    F_total_edges[self.mesh['edges_index_inside_triangle'][index][2]]+=F_element[5]  


                if (self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]==2):
                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[3]=1.0
                    det=np.sqrt(np.square(vertices[0,0]-vertices[1,0]) +np.square(vertices[0,1]-vertices[1,1]))
                    points_edge1=(B@points_edge1_ref.T +c).T

                    res=self.pb.neumann(points_edge1[:,0],points_edge1[:,1])
                    res=np.reshape(res,(-1,1))
                    test=self.b.interpolate(points_edge1_ref,select)
                    print('writing1 --> ',self.mesh['vertex_markers'][triangle[2]],triangle)
                    F_total_edges[self.mesh['edges_index_inside_triangle'][index][0]]+=det*np.sum(neumann_weights*res*test)
                    print(test,det,res)


                if (self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]==2):
                    print('writing2 --> ',self.mesh['vertex_markers'][triangle[2]],triangle)

                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[4]=1.0
                    det= np.sqrt(np.square(vertices[1,0]-vertices[2,0]) +np.square(vertices[1,1]-vertices[2,1]))
                    points_edge2=(B@points_edge1_ref.T +c).T

                    res=self.pb.neumann(points_edge2[:,0],points_edge2[:,1])
                    res=np.reshape(res,(-1,1))
                    test=self.b.interpolate(points_edge2_ref,select)
                    
                    F_total_edges[self.mesh['edges_index_inside_triangle'][index][1]]+=det*np.sum(neumann_weights*res*test)
                    print(test,det,res)

                if (self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]==2):
                    print('writing3 --> ',self.mesh['vertex_markers'][triangle[2]],triangle)
                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[5]=1.0
                    det= np.sqrt(np.square(vertices[0,0]-vertices[2,0]) +np.square(vertices[0,1]-vertices[2,1]))
                    points_edge3=(B@points_edge1_ref.T +c).T

                    res=self.pb.neumann(points_edge3[:,0],points_edge3[:,1])
                    res=np.reshape(res,(-1,1))
                    test=self.b.interpolate(points_edge3_ref,select)
                    print(test,det,res)
                    F_total_edges[self.mesh['edges_index_inside_triangle'][index][2]]+=det*np.sum(neumann_weights*res*test)

            xy_quad_total.append(xy_quad_element)

        self.J_total=J_total
        self.F_total_vertices=tf.constant(F_total_vertices,dtype=tf.float64)
        self.F_total_edges=tf.constant(F_total_edges,dtype=tf.float64)
        xy_quad_total = np.array(xy_quad_total,dtype=np.float64)
        self.xy_quad_total=tf.constant(xy_quad_total,dtype=tf.float64)

        self.mu = tf.constant(mu, dtype=tf_type)
        self.sigma = tf.constant(sigma, dtype=tf_type)
        self.beta = tf.constant(beta, dtype=tf_type)

        print(tf.shape(self.mu), tf.shape(self.beta), tf.shape(self.sigma))
        
    def summary(self):
        print('-->mesh : ')
        print('     n_triangles : ',self.n_triangles)
        print('     n_vertices  : ',self.n_vertices)
        print('     n_edges     : ',self.n_edges)
        print('     h_max           : ',self.mesh['h_max'])
        print('     h_min           : ',self.mesh['h_min'])
        print('-->test_fun      : ')
        print('     order       : ',self.params['N_test'])
        print('     dof         : ',self.dof)