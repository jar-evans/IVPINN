from my_types import *
from mesh import find_hs

from interpolator import *
import time
from quad import *


class VPINN():

    def __init__(self, pb, params, mesh,verbose,NN = None):

        super().__init__()

        # accept parameters
        self.pb = pb
        self.params = params
        self.n_test = params['N_test']

        self.mesh = mesh

        self.n_vertices=len(mesh['vertices'])

        self.n_edges=len(mesh['edges'])

        self.n_triangles=len(mesh['triangles'])

        self.verbose=verbose 

        if self.n_test>=2:
            self.dof=(len(self.mesh['vertex_markers'][self.mesh['vertex_markers']!=1.0]))+(len(self.mesh['edge_markers'][self.mesh['edge_markers']!=1.0]))
        else:
            self.dof=(len(self.mesh['vertex_markers'][self.mesh['vertex_markers']!=1.0]))

        self.pre_compute()

        # add the neural network to the class if given at initialisation
        if NN:
            self.set_NN(NN)

        self.sum_of_vectors_vertices = tf.Variable(tf.zeros((self.n_vertices,1),dtype=tf_type)) 
        self.sum_of_vectors_vertices.assign(tf.zeros_like(self.sum_of_vectors_vertices))   

        if self.n_test>=2:
            self.sum_of_vectors_edges = tf.Variable(tf.zeros((self.n_edges,1),dtype=tf_type))       
            self.sum_of_vectors_edges.assign(tf.zeros_like(self.sum_of_vectors_edges))

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
    
    @tf.function
    def eval_grad_NN(self,x):
        """input tensor of size (n,2) returns tensor of size (n,2) ->on each row you have first der x and then der y"""
        with tf.GradientTape() as tape:
            tape.watch(x)
            res=self.NN(x)
        grad=tape.gradient(res,x)
        return grad

    @tf.function
    def eval_laplacian_NN(self,x):
        """input tensor of size (n,2) returns tensor of size (n,2) ->on each row you have first derder x and then derder y"""
        with tf.GradientTape() as tape_:
            #tape_.watch(x) here 
            with tf.GradientTape() as tape:
                tape.watch(x)
                res=self.NN(x)
            grad=tape.gradient(res,x)

            tape_.watch(x) #or here 
            laplacian=tape_.gradient(grad,x)
        return laplacian
    
    def eval_NN_and_grad(self,x):
        """input tensor of size (n,2) returns tensor eval of size (n,2)  + tensor grad of size (n,2)"""
        with tf.GradientTape() as tape:
            tape.watch(x)
            res=self.NN(x)
        grad=tape.gradient(res,x)
        return res,grad

    
    def u_NN(self, x, y):
        eval=tf.constant([[x,y]],dtype=tf.float64)
        return self.NN(eval).numpy()[0,0] 
    
    def u_NN_BC(self, x, y):
        """for numpy"""
        eval=tf.constant([[x,y]],dtype=tf.float64)
        return self.NN_imposeBC(eval).numpy()[0,0] 

    def boundary_function(self,x):
        return tf.expand_dims(x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1]),axis=1)

    def NN_imposeBC(self,x):
        eval=self.NN(x)
        boundary=self.boundary_function(x)
        return eval*boundary + self.bc_model(x)
    
    # def u_NN_BC_grad(self, x):
    #     eval=tf.constant(x,dtype=tf_type)
    #     d1, d2 = self.eval_grad_NN_BC(eval)[0,:]
    #     return np.array([d1, d2], dtype=np_type)

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

    @tf.function
    def boundary_loss(self):
    ## NOTE:impose boundary or same structure for ICs
        prediction = self.eval_NN(self.boundary_points)
        u_bound_exact=self.u_bound_exact
        return tf.reduce_mean(tf.square(u_bound_exact - prediction))
    
    def set_bc_model(self, bc_model):
        self.bc_model = bc_model


    def standard_bc(self,x):
        return tf.expand_dims((tf.sin(x[:,0])*tf.sin(1-x[:,0])*tf.sin(x[:,1])*tf.sin(1-x[:,1])+1.0)*self.pb.u_exact(x[:,0],x[:,1]),axis=1)

    @tf.function
    def custom_loss(self):
 
        n_triangles=self.n_triangles
        xy_quad_total =self.xy_quad_total
        w_quad = tf.concat([self.w_quad.T,self.w_quad.T], axis=0)
        r=self.n_test

        #eval in one shot 
        x_eval=tf.reshape(xy_quad_total,(-1,2))

        grad=self.eval_grad_NN_BC(x_eval)
        grad_=tf.reshape(grad,(n_triangles,-1,2))

        F_total_vertices=self.F_total_vertices
        sum_of_vectors_vertices=self.sum_of_vectors_vertices

        if r > 1:
            F_total_edges=self.F_total_edges   
            sum_of_vectors_edges=self.sum_of_vectors_edges

        grad_test=self.grad_test  

        for index,triangle in enumerate(self.mesh['triangles']):

            grad_elem=tf.transpose(grad_[index])   

            # get quadrature points in arb. element and get jacobian
            B,c,J,B_D,B_DD=self.b.change_of_coordinates(self.mesh['vertices'][triangle])

            grad_test_elem=B_D @ grad_test
      
            v0= J*tf.reduce_sum(w_quad*grad_test_elem[0]*grad_elem)
            v1= J*tf.reduce_sum(w_quad*grad_test_elem[1]*grad_elem)
            v2= J*tf.reduce_sum(w_quad*grad_test_elem[2]*grad_elem)

            if (self.mesh['vertex_markers'][triangle[0]]==1):
                v0=v0*0.0
            if (self.mesh['vertex_markers'][triangle[1]]==1):
                v1=v1*0.0
            if (self.mesh['vertex_markers'][triangle[2]]==1):
                v2=v2*0.0

            indices_vertices = tf.constant([[triangle[0]], [triangle[1]], [triangle[2]]])
            updates_vertices = [v0, v1, v2]

            scatter_vertices = tf.scatter_nd(indices_vertices, updates_vertices, [self.n_vertices])
            scatter_vertices_=tf.expand_dims(scatter_vertices,axis=1)
            sum_of_vectors_vertices = tf.reduce_sum([sum_of_vectors_vertices, scatter_vertices_], axis=0)

            if r > 1:

                l0=J*tf.reduce_sum(w_quad*grad_test_elem[3]*grad_elem)
                l1=J*tf.reduce_sum(w_quad*grad_test_elem[4]*grad_elem)
                l2=J*tf.reduce_sum(w_quad*grad_test_elem[5]*grad_elem)

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]==1):
                    l0=l0*0.0

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]==1):
                    l1=l1*0.0

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]==1):
                    l2=l2*0.0
                
                indices_edges = tf.constant([[self.mesh['edges_index_inside_triangle'][index][0]], [self.mesh['edges_index_inside_triangle'][index][1]], [self.mesh['edges_index_inside_triangle'][index][2]]])
                updates_edges=[l0,l1,l2]

                scatter_edges = tf.scatter_nd(indices_edges, updates_edges, [self.n_edges])
                scatter_edges_=tf.expand_dims(scatter_edges,axis=1)    
                sum_of_vectors_edges = tf.reduce_sum([sum_of_vectors_edges, scatter_edges_], axis=0)


        #tf.reduce_sum(a_vertices,axis=1,keepdims=True)-F_total_vertices,tf.reduce_sum(a_edges,axis=1,keepdims=True)-F_total_edges
        if r > 1:
            return (tf.reduce_sum(tf.square(sum_of_vectors_vertices-F_total_vertices))+tf.reduce_sum(tf.square(sum_of_vectors_edges-F_total_edges)))/self.dof
        else:
            return (tf.reduce_sum(tf.square(sum_of_vectors_vertices-F_total_vertices)))/self.dof


    def calc_residuals(self):

        n_triangles=self.n_triangles
        xy_quad_total =self.xy_quad_total
        w_quad = tf.concat([self.w_quad.T,self.w_quad.T], axis=0)
        r=self.n_test

        #eval in one shot 
        x_eval=tf.reshape(xy_quad_total,(-1,2))

        grad=self.eval_grad_NN_BC(x_eval)
        grad_=tf.reshape(grad,(n_triangles,-1,2))

        F_total_vertices=self.F_total_vertices
        sum_of_vectors_vertices=self.sum_of_vectors_vertices

        if r > 1:
            F_total_edges=self.F_total_edges   
            sum_of_vectors_edges=self.sum_of_vectors_edges

        grad_test=self.grad_test  

        for index,triangle in enumerate(self.mesh['triangles']):

            grad_elem=tf.transpose(grad_[index])   

            # get quadrature points in arb. element and get jacobian
            B,c,J,B_D,B_DD=self.b.change_of_coordinates(self.mesh['vertices'][triangle])

            grad_test_elem=B_D @ grad_test
      
            v0= J*tf.reduce_sum(w_quad*grad_test_elem[0]*grad_elem)
            v1= J*tf.reduce_sum(w_quad*grad_test_elem[1]*grad_elem)
            v2= J*tf.reduce_sum(w_quad*grad_test_elem[2]*grad_elem)

            if (self.mesh['vertex_markers'][triangle[0]]==1):
                v0=v0*0.0
            if (self.mesh['vertex_markers'][triangle[1]]==1):
                v1=v1*0.0
            if (self.mesh['vertex_markers'][triangle[2]]==1):
                v2=v2*0.0

            indices_vertices = tf.constant([[triangle[0]], [triangle[1]], [triangle[2]]])
            updates_vertices = [v0, v1, v2]

            scatter_vertices = tf.scatter_nd(indices_vertices, updates_vertices, [self.n_vertices])
            scatter_vertices_=tf.expand_dims(scatter_vertices,axis=1)
            sum_of_vectors_vertices = tf.reduce_sum([sum_of_vectors_vertices, scatter_vertices_], axis=0)

            if r > 1:

                l0=J*tf.reduce_sum(w_quad*grad_test_elem[3]*grad_elem)
                l1=J*tf.reduce_sum(w_quad*grad_test_elem[4]*grad_elem)
                l2=J*tf.reduce_sum(w_quad*grad_test_elem[5]*grad_elem)

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]==1):
                    l0=l0*0.0

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]==1):
                    l1=l1*0.0

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]==1):
                    l2=l2*0.0
                
                indices_edges = tf.constant([[self.mesh['edges_index_inside_triangle'][index][0]], [self.mesh['edges_index_inside_triangle'][index][1]], [self.mesh['edges_index_inside_triangle'][index][2]]])
                updates_edges=[l0,l1,l2]

                scatter_edges = tf.scatter_nd(indices_edges, updates_edges, [self.n_edges])
                scatter_edges_=tf.expand_dims(scatter_edges,axis=1)    
                sum_of_vectors_edges = tf.reduce_sum([sum_of_vectors_edges, scatter_edges_], axis=0)


        #tf.reduce_sum(a_vertices,axis=1,keepdims=True)-F_total_vertices,tf.reduce_sum(a_edges,axis=1,keepdims=True)-F_total_edges
        if r > 1:
            return tf.square(sum_of_vectors_vertices-F_total_vertices), tf.square(sum_of_vectors_edges-F_total_edges)
        else:
            return tf.square(sum_of_vectors_vertices-F_total_vertices)

        #tf.reduce_sum(a_vertices,axis=1,keepdims=True)-F_total_vertices,tf.reduce_sum(a_edges,axis=1,keepdims=True)-F_total_edges
        # return tf.square(sum_of_vectors_vertices-F_total_vertices), tf.square(sum_of_vectors_edges-F_total_edges)

    @tf.function
    def loss_total(self):
        res=self.custom_loss()
        return res 
    
    #@tf.function
    def loss_gradient(self):       
        with tf.GradientTape() as tape:
            loss = self.loss_total()

        gradient = tape.gradient(loss, self.NN.trainable_variables)
        return loss, gradient

    @tf.function
    def gradient_descent(self):
        loss, gradient = self.loss_gradient()
        self.optimizer.apply_gradients(zip(gradient, self.NN.trainable_variables))
        return loss

    def train(self, iter,LR ,bc_model=None):

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)



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
        #for ivpinns
        #self.B=interpolator(2,False,False,points=None)

        grad=[]

        for i in range(self.b.n):
            elem=np.stack([self.b.Base_dx[:,i],self.b.Base_dy[:,i]])
            grad.append(elem)

        self.grad_test=tf.constant(grad)

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


    def construct_RHS(self):
        #modify also this 

        r=self.b.r
        F_total_vertices = np.zeros((self.n_vertices,1),dtype=np_type)

        if r>=2:
            F_total_edges = np.zeros((self.n_edges,1),dtype=np_type)
        
        xy_quad_total = []
        
        for index,triangle in enumerate(self.mesh['triangles']):
            F_element=[]
            x_quad=self.points

            # get quadrature points in arb. element and get jacobian
            B,c,J,B_D,B_DD=self.b.change_of_coordinates(self.mesh['vertices'][triangle])

            xy_quad_element=(B@x_quad.T +c).T
      
            # evaluate f on arb. quad points
            f_quad_element = self.pb.f_exact(xy_quad_element[:, 0], xy_quad_element[:, 1])

            # do the integral and appnd to total list
            F_element=[J*np.sum(self.w_quad[:,0]*self.b.Base[:,r]*f_quad_element) for r in range(self.b.n)]
            F_element=np.array(F_element,dtype=np_type)

            if (self.mesh['vertex_markers'][triangle[0]]==0):
                F_total_vertices[triangle[0]]+=F_element[0]

            if (self.mesh['vertex_markers'][triangle[1]]==0):
                F_total_vertices[triangle[1]]+=F_element[1]
 
            if (self.mesh['vertex_markers'][triangle[2]]==0):
                F_total_vertices[triangle[2]]+=F_element[2]

            if(r>=2):
                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]==0):
                    F_total_edges[self.mesh['edges_index_inside_triangle'][index][0]]+=F_element[3]

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]==0):
                        F_total_edges[self.mesh['edges_index_inside_triangle'][index][1]]+=F_element[4]

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]==0):
                    F_total_edges[self.mesh['edges_index_inside_triangle'][index][2]]+=F_element[5]  


            xy_quad_total.append(xy_quad_element)
    
        self.F_total_vertices=tf.constant(F_total_vertices,dtype=tf_type)

        if r>=2:
            self.F_total_edges=tf.constant(F_total_edges,dtype=tf_type)

        xy_quad_total = np.array(xy_quad_total,dtype=np_type)
        self.xy_quad_total=tf.constant(xy_quad_total,dtype=tf_type)

    def summary(self):
        h_s=find_hs(self.mesh)
        print('-->mesh : ')
        print('     n_triangles : ',self.n_triangles)
        print('     n_vertices  : ',self.n_vertices)
        print('     n_edges     : ',self.n_edges)
        print('     h_max           : ',h_s[0])
        print('     h_min           : ',h_s[1])
        print('-->test_fun      : ')
        print('     order       : ',self.params['N_test'])
        print('     dof         : ',self.dof)