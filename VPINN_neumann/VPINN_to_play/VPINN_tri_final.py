from my_types import *
from mesh import find_hs

from pyDOE import lhs

from interpolator import *
import time
from quad import *



#tf.compat.v1.disable_eager_execution()



class VPINN(tf.keras.Model):

    def __init__(self, pb, params, mesh, NN = None):

        super().__init__()

        # accept parameters
        self.pb = pb
        self.params = params
        self.n_test = params['N_test']

        self.mesh = mesh
        self.n_vertices=len(mesh['vertices'])
        self.n_edges=len(mesh['edges'])

        self.n_triangles=len(mesh['triangles'])
        if self.n_test>=2:
            self.dof=(len(self.mesh['vertex_markers'][self.mesh['vertex_markers']!=1.0]))+(len(self.mesh['edge_markers'][self.mesh['edge_markers']!=1.0]))
        else:
            self.dof=(len(self.mesh['vertex_markers'][self.mesh['vertex_markers']!=1.0]))




        self.pre_compute()

        # add the neural network to the class if given at initialisation
        if NN:
            self.set_NN(NN)


        self.summary()

    def set_NN(self, NN, LR=0.001):


        self.NN = NN

        # take trainable vars
        self.vars = self.NN.trainable_variables

        # set optimiser
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)





    @tf.function
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

    

    


    def u_NN_BC______________(self, x, y):
        eval=tf.constant([[x,y]],dtype=tf.float64)
        return self.NN_imposeBC(eval).numpy()[0,0] 





    def boundary_function(self,x):
        return tf.expand_dims(x[:,1]*(1-x[:,1]),axis=1)



    def dirichlet(self,x):
        return tf.expand_dims((1-x[:,1])*tf.cos(np.pi*x[:,0])+ x[:,1]*tf.cos(np.pi*x[:,0])*(1/np.e),axis=1)     


    def NN_imposeBC(self,x):
        eval=self.NN(x)
        boundary=self.boundary_function(x)
        return eval*boundary + self.dirichlet(x)
    

    @tf.function
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
    
    def eval_grad_NN_BC__(self,x):
        """
        input tensor of size (n,2)
        returns tensor of size (n,2) -> on each row you have first der x and then der y
        """

        with tf.GradientTape() as tape:
            tape.watch(x)
            res=self.NN_imposeBC(x)
            
        grad=tape.gradient(res,x)
        return grad.numpy()




    @tf.function
    def custom_loss(self):
        #a_vertices = self.a_vertices #
        #a_edges = self.a_edges #

        n_triangles=self.n_triangles
        xy_quad_total =self.xy_quad_total
        dof=self.dof

        w_quad = tf.concat([self.w_quad.T,self.w_quad.T], axis=0)


        #eval in one shot 
        x_eval=tf.reshape(xy_quad_total,(-1,2))

        grad=self.eval_grad_NN_BC(x_eval)
        grad_=tf.reshape(grad,(n_triangles,-1,2))

        F_total_vertices=self.F_total_vertices
        
        if self.n_test>=2:
            F_total_edges=self.F_total_edges

   

        grad_test=self.grad_test  

        #sum_of_vectors_vertices = tf.Variable(tf.zeros((self.n_vertices,1),dtype=tf.float64))
        #sum_of_vectors_edges = tf.Variable(tf.zeros((self.n_edges,1),dtype=tf.float64))
        sum_of_vectors_vertices=self.sum_of_vectors_vertices
        if self.n_test>=2:
            sum_of_vectors_edges=self.sum_of_vectors_edges



        for index,triangle in enumerate(self.mesh['triangles']):



            grad_elem=tf.transpose(grad_[index])
   

            # get quadrature points in arb. element and get jacobian
            B,c,J,B_D,B_DD=self.b.change_of_coordinates(self.mesh['vertices'][triangle])

            grad_test_elem=B_D @ grad_test
      
            v0= J*tf.reduce_sum(w_quad*grad_test_elem[0]*grad_elem)
            v1= J*tf.reduce_sum(w_quad*grad_test_elem[1]*grad_elem)
            v2= J*tf.reduce_sum(w_quad*grad_test_elem[2]*grad_elem)

            if self.n_test>=2:
                l0=J*tf.reduce_sum(w_quad*grad_test_elem[3]*grad_elem)
                l1=J*tf.reduce_sum(w_quad*grad_test_elem[4]*grad_elem)
                l2=J*tf.reduce_sum(w_quad*grad_test_elem[5]*grad_elem)

            if (self.mesh['vertex_markers'][triangle[0]]==1):
                v0=v0*0.0


            if (self.mesh['vertex_markers'][triangle[1]]==1):
                v1=v1*0.0
 

            if (self.mesh['vertex_markers'][triangle[2]]==1):
                v2=v2*0.0






            if self.n_test>=2:
                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]==1):
                    l0=l0*0.0

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]==1):
                    l1=l1*0.0

                if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]==1):
                    l2=l2*0.0
            







            indices_vertices = tf.constant([[triangle[0]], [triangle[1]], [triangle[2]]])
            if self.n_test>=2:
                indices_edges = tf.constant([[self.mesh['edges_index_inside_triangle'][index][0]], [self.mesh['edges_index_inside_triangle'][index][1]], [self.mesh['edges_index_inside_triangle'][index][2]]])

            
            updates_vertices = [v0, v1, v2]
            if self.n_test>=2:
                updates_edges=[l0,l1,l2]

            #sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

            #tf.sparse.add()




            scatter_vertices = tf.scatter_nd(indices_vertices, updates_vertices, [self.n_vertices])
            scatter_vertices_=tf.expand_dims(scatter_vertices,axis=1)

            if self.n_test>=2:
                scatter_edges = tf.scatter_nd(indices_edges, updates_edges, [self.n_edges])
                scatter_edges_=tf.expand_dims(scatter_edges,axis=1)
            
        
            sum_of_vectors_vertices = tf.reduce_sum([sum_of_vectors_vertices, scatter_vertices_], axis=0)

            if self.n_test>=2:
                sum_of_vectors_edges = tf.reduce_sum([sum_of_vectors_edges, scatter_edges_], axis=0)

            #a_vertices[triangle[0]]+=a_element[0]

        
            

        


        #tf.reduce_sum(a_vertices,axis=1,keepdims=True)-F_total_vertices,tf.reduce_sum(a_edges,axis=1,keepdims=True)-F_total_edges
        if self.n_test>=2:
            return (tf.reduce_sum(tf.square(sum_of_vectors_vertices-F_total_vertices))+tf.reduce_sum(tf.square(sum_of_vectors_edges-F_total_edges)))/self.dof
        else:
            return tf.reduce_sum(tf.square(sum_of_vectors_vertices-F_total_vertices))/self.dof


    @tf.function
    def loss_total(self):
        #loss_b = self.boundary_loss()
        #res = self.variational_loss(tape)
        res=self.custom_loss()
        return  res #+loss_b
    
    #@tf.function
    def loss_gradient(self):

       # self.a_vertices.assign(tf.zeros((self.n_vertices,self.n_triangles),dtype=tf.float64))
       # self.a_edges.assign(tf.zeros((self.n_edges,self.n_triangles),dtype=tf.float64))
        
        with tf.GradientTape() as tape:
            #loss_grad.watch(self.xy_quad_total)
            #tape.watch(self.NN.trainable_variables)
            loss = self.loss_total()
            #print(loss)
            #loss=(tf.reduce_sum(tf.square(res_vertices))+tf.reduce_sum(tf.square(res_edges)))/self.dof

        gradient = tape.gradient(loss, self.NN.trainable_variables)
        return loss, gradient

    @tf.function
    def gradient_descent(self):
        loss, gradient = self.loss_gradient()
        
        self.optimizer.apply_gradients(zip(gradient, self.NN.trainable_variables))
        return loss

    def train(self, iter,LR):


        
        #self.a_vertices = tf.Variable(tf.zeros((self.n_vertices,self.n_triangles),dtype=tf.float64)) #
        #self.a_edges = tf.Variable(tf.zeros((self.n_edges,self.n_triangles),dtype=tf.float64)) #


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

        history = []


        self.sum_of_vectors_vertices = tf.Variable(tf.zeros((self.n_vertices,1),dtype=tf.float64))
        if self.n_test>=2: #
            self.sum_of_vectors_edges = tf.Variable(tf.zeros((self.n_edges,1),dtype=tf.float64))       #

        start_time = time.time()
        for i in range(iter+1):
            self.sum_of_vectors_vertices.assign(tf.zeros_like(self.sum_of_vectors_vertices))
            if self.n_test>=2:       #
                self.sum_of_vectors_edges.assign(tf.zeros_like(self.sum_of_vectors_edges))             #


            loss = self.gradient_descent()  #add other losses 
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f'Iteration: {i}', f'loss: {loss.numpy():0.15f}', f'time: {elapsed}')

                history.append(loss)
                start_time = time.time()
                

        return history

    def get_domain_info(self):

        a = np.array(self.params['domain'][0])
        b = np.array(self.params['domain'][1])

        scale = b - a
        mid = (a + b)*0.5

        return a, b, scale, mid



    def generate_boundary_points(self):
        # Boundary points
        a = (0, 0)
        b = (1, 0)
        c = (1, 1)
        d = (0, 1)


        boundary_points = self.generate_rectangle_points(a, b, c, d, self.params['n_bound'])

        self.boundary_points=tf.constant(boundary_points,dtype=tf.float64)


        u_bound_exact = self.pb.u_exact(self.boundary_points[:, 0], self.boundary_points[:, 1])
        self.u_bound_exact=tf.reshape(u_bound_exact,(-1,1))





    # def generate_inner_points(self):
    #     _, _, scale, mid = self.get_domain_info()

    #     #NOTE: probably using wrong constant here
    #     temp = np.array(scale*(lhs(2, self.params['n_bound']) - 0.5) + mid)

    #     self.xf = temp[:, 0]
    #     self.yf = temp[:, 1]
    #     self.X_f_train = np.hstack((self.xf[:, None], self.yf[:, None]))

    #     ff = np.asarray([self.pb.f_exact(self.xf[j], self.yf[j])
    #                     for j in range(len(self.yf))])
    #     self.f_train = ff[:, None]

    def pre_compute(self):


        self.generate_quadrature_points()

        self.evaluate_test_and_inter_functions()

        self.construct_RHS()



    def evaluate_test_and_inter_functions(self):
    
        self.b=interpolator(self.params['N_test'],False,True,points=self.points)

        self.B=interpolator(2,False,False,points=None)

        grad=[]

        for i in range(self.b.n):
            elem=np.stack([self.b.Base_dx[:,i],self.b.Base_dy[:,i]])
            grad.append(elem)

        self.grad_test=tf.constant(grad)

    def generate_points_on_edge(self,point1, point2, num_points):
        x_vals = np.linspace(point1[0], point2[0], num_points,endpoint=False)
        y_vals = np.linspace(point1[1], point2[1], num_points,endpoint=False)
        points_on_edge = np.column_stack((x_vals, y_vals))
        return points_on_edge

    def generate_rectangle_points(self,a, b, c, d, num_points_per_edge):

        edge1 = self.generate_points_on_edge(a, b, num_points_per_edge)
        edge2 = self.generate_points_on_edge(b, c, num_points_per_edge)
        edge3 = self.generate_points_on_edge(c, d, num_points_per_edge)
        edge4 = self.generate_points_on_edge(d, a, num_points_per_edge)

        rectangle_points = np.vstack((edge1, edge2, edge3, edge4))
        return rectangle_points    


    def generate_quadrature_points(self):
        """
        here you will have col vectors with correct stuff defined on the ref triangle
        self.x_quad
        self.y_quad 
        self.w_quad
        """
        self.xy_quad =np.array(points,dtype=np.float64)
        self.w_quad = np.array(weights,dtype=np.float64)/2.0
        self.x_quad = np.reshape(self.xy_quad[:,0], (len(self.xy_quad), 1))
        self.y_quad = np.reshape(self.xy_quad[:,1], (len(self.xy_quad), 1))

        self.points=self.xy_quad
        
        self.w_quad = np.reshape(self.w_quad, (len(self.w_quad), 1))

    def pre_compute(self):
        self.generate_quadrature_points()
        self.evaluate_test_and_inter_functions()
        self.construct_RHS()

    def evaluate_test_functions(self):
        self.v_evaluations = {}
        self.v_evaluations["vx_quad"] = self.pb.v(self.x_quad, self.y_quad, self.n_test)
        self.n_test = np.shape(self.v_evaluations["vx_quad"])[0]
        # print(self.n_test)
        # self.v_evaluations["dv_x_quad"], self.v_evaluations["d2v_x_quad"] = self.pb.dtest_func(self.n_test, self.x_quad)
        # self.v_evaluations["vy_quad"] = self.pb.v_y(self.n_test, self.y_quad)
        # self.v_evaluations["dv_y_quad"], self.v_evaluations["d2v_y_quad"] = self.pb.dtest_func(self.n_test, self.y_quad)
        # print(np.max(self.v_evaluations["v_x_quad"]), np.min(self.v_evaluations["v_x_quad"]), np.sum(self.v_evaluations["v_x_quad"]))
        # print(np.max(self.v_evaluations["v_y_quad"]), np.min(self.v_evaluations["v_y_quad"]), np.sum(self.v_evaluations["v_y_quad"]))

    def construct_RHS(self):
        #modify also this 


        # vx_quad = self.v_evaluations["vx_quad"]
        # vy_quad = self.v_evaluations["vy_quad"]

        F_total_vertices = np.zeros((self.n_vertices,1),dtype=np.float64)
        F_total_edges = np.zeros((self.n_edges,1),dtype=np.float64)
        r=self.b.r
        

        xy_quad_total = []
        J_total = []
        F_total=[]
        
        neumann_points_ref=self.b.neumann_nodes
        neumann_weights=self.b.neumannn_weights



        points_edge1_ref=np.hstack([neumann_points_ref,np.zeros_like(neumann_points_ref)],dtype=np_type)
        points_edge2_ref=np.hstack([1.0-neumann_points_ref,neumann_points_ref],dtype=np_type)
        points_edge3_ref=np.hstack([np.zeros_like(neumann_points_ref),neumann_points_ref[::-1]],dtype=np_type)
 

        for index,triangle in enumerate(self.mesh['triangles']):
            F_element=[]
            x_element=[]
            J_element=[]
            x_quad=self.points

            vertices=self.mesh['vertices'][triangle] #3x2 


            # get quadrature points in arb. element and get jacobian
            B,c,J,B_D,B_DD=self.b.change_of_coordinates(self.mesh['vertices'][triangle])

            xy_quad_element=(B@x_quad.T +c).T
      
            


            J_element.append(J)

            # evaluate f on arb. quad points
            f_quad_element = self.pb.f_exact(xy_quad_element[:, 0], xy_quad_element[:, 1])

            # do the integral and appnd to total list
            #print([J*np.sum(self.w_quad[:,0]*self.b.Base[:,r]*f_quad_element) for r in range(self.b.n)])
            F_element=[J*np.sum(self.w_quad[:,0]*self.b.Base[:,r]*f_quad_element) for r in range(self.b.n)]

            

            F_element=np.array(F_element,dtype=np_type)
            J_element=np.array(J_element,dtype=np_type)


            # if (self.mesh['vertex_markers'][triangle[0]]!=1):
            #         F_total_vertices[triangle[0]]+=F_element[0]


            # if (self.mesh['vertex_markers'][triangle[1]]!=1):
            #         F_total_vertices[triangle[1]]+=F_element[1]
 

            # if (self.mesh['vertex_markers'][triangle[2]]!=1):
            #         F_total_vertices[triangle[2]]+=F_element[2]



            #neumann 
            if (self.mesh['vertex_markers'][triangle[0]]==2 and self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]==2):
                    
                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[0]=1.0
                    det=np.sqrt(np.square(vertices[0,0]-vertices[1,0]) +np.square(vertices[0,1]-vertices[1,1]))
                    points_edge1=(B@points_edge1_ref.T +c).T

                    res=self.pb.neumann(points_edge1[:,0],points_edge1[:,1])
                    res=np.reshape(res,(-1,1))
                    test=self.b.interpolate(points_edge1_ref,select)

                    F_total_vertices[triangle[0]]+=det*np.sum(neumann_weights*res*test)
    

            if (self.mesh['vertex_markers'][triangle[1]]==2 and self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]==2):

                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[1]=1.0
                    det= np.sqrt(np.square(vertices[1,0]-vertices[2,0]) +np.square(vertices[1,1]-vertices[2,1]))
                    points_edge2=(B@points_edge1_ref.T +c).T

                    res=self.pb.neumann(points_edge2[:,0],points_edge2[:,1])
                    res=np.reshape(res,(-1,1))
                    test=self.b.interpolate(points_edge2_ref,select)
                    

                    F_total_vertices[triangle[1]]+=det*np.sum(neumann_weights*res*test)

        

            if (self.mesh['vertex_markers'][triangle[2]]==2 and self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]==2):
                

                    select=np.zeros((self.b.n,1),dtype=np_type)
                    select[2]=1.0
                    det= np.sqrt(np.square(vertices[0,0]-vertices[2,0]) +np.square(vertices[0,1]-vertices[2,1]))
                    points_edge3=(B@points_edge1_ref.T +c).T

                    res=self.pb.neumann(points_edge3[:,0],points_edge3[:,1])
                    res=np.reshape(res,(-1,1))
                    test=self.b.interpolate(points_edge3_ref,select)
                    

                    F_total_vertices[triangle[2]]+=det*np.sum(neumann_weights*res*test)
                



            if(r>=2):
                    # if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]!=1):
                    #     F_total_edges[self.mesh['edges_index_inside_triangle'][index][0]]+=F_element[3]

                    # if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]!=1):
                    #         F_total_edges[self.mesh['edges_index_inside_triangle'][index][1]]+=F_element[4]

                    # if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]!=1):
                    #     F_total_edges[self.mesh['edges_index_inside_triangle'][index][2]]+=F_element[5]  


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
            #print(F_total_edges)

            xy_quad_total.append(xy_quad_element)
    
            #J_total.append(J)


        self.J_total=J_total
        self.F_total_vertices=tf.constant(F_total_vertices,dtype=tf.float64)
        self.F_total_edges=tf.constant(F_total_edges,dtype=tf.float64)
        xy_quad_total = np.array(xy_quad_total,dtype=np.float64)
        self.xy_quad_total=tf.constant(xy_quad_total,dtype=tf.float64)

        #print(np.shape(self.w_quad[:,0]),np.shape(self.b.Base[:,0]))


        #print(self.F_total_edges)
        #print(self.F_total_vertices)
        #print(self.xy_quad_total)
        #print(tf.shape(self.xy_quad_total))


        #add another loop for the neumann 
        #add also the neumann condition in the pb


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






    def plot_loss_history(self, loss_history, PLOT):
        fontsize = 24
        fig = plt.figure(1)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
        plt.xlabel('$iteration$', fontsize=fontsize)
        plt.ylabel('$loss \,\, values$', fontsize=fontsize)
        plt.yscale('log')
        plt.grid(True)
        plt.plot(loss_history)
        plt.tick_params(labelsize=20)
        # fig.tight_layout()
        fig.set_size_inches(w=11, h=11)

        if PLOT == 'save':
            plt.savefig('VPINN_loss_history.pdf')
        else:
            plt.show()
        

