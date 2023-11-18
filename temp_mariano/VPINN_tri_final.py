import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights
from interpolator import *
import time
SEED = 42
from quad import *

#tf.compat.v1.disable_eager_execution()



class VPINN(tf.keras.Model):

    def __init__(self, pb, params, mesh, NN = None):

        super().__init__()

        # accept parameters
        self.pb = pb
        self.params = params
        self.n_test = params['n_test']

        self.mesh = mesh
        self.n_vertices=len(mesh['vertices'])
        self.n_edges=len(mesh['edges'])
        self.n_triangles=len(mesh['triangles'])
        self.dof=(self.n_vertices-np.sum(self.mesh['vertex_markers']))+(self.n_edges-np.sum(self.mesh['edge_markers']))
        print(self.dof)

        self.n_el_x, self.n_el_y = self.params['n_elements']

        # generate all points/coordinates to be used in the process
        self.generate_boundary_points()
        # self.generate_inner_points()
        self.pre_compute()

        # add the neural network to the class if given at initialisation
        if NN:
            self.set_NN(NN)

    def set_NN(self, NN, LR=0.01):
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        # initialise the NN
        self.NN = NN

        # take trainable vars
        self.vars = self.NN.trainable_variables

        # set optimiser
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)



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

    
    def u_NN(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        y = tf.convert_to_tensor(y, dtype=tf.float64)  
        return self.NN(tf.concat([x, y], 1))


    @tf.function
    def boundary_loss(self):
    ## NOTE:impose boundary or same structure for ICs
        prediction = self.eval_NN(self.boundary_points)
        u_bound_exact=self.u_bound_exact
        return tf.reduce_mean(tf.square(u_bound_exact - prediction))

    
    def custom_loss(self):
        #a_vertices = self.a_vertices #
        #a_edges = self.a_edges #

        n_triangles=self.n_triangles
        xy_quad_total =self.xy_quad_total
        dof=self.dof

        w_quad = tf.concat([self.w_quad.T,self.w_quad.T], axis=0)


        #eval in one shot 
        x_eval=tf.reshape(xy_quad_total,(-1,2))

        grad=self.eval_grad_NN(x_eval)
        grad_=tf.reshape(grad,(n_triangles,-1,2))

        F_total_vertices=self.F_total_vertices
        F_total_edges=self.F_total_edges

   

        grad_test=self.grad_test  

        sum_of_vectors = tf.Variable(tf.zeros((self.n_vertices,1),dtype=tf.float64))
        


        for index,triangle in enumerate(self.mesh['triangles']):



            grad_elem=tf.transpose(grad_[index])
   

            # get quadrature points in arb. element and get jacobian
            B,c,J,B_D,B_DD=self.b.change_of_coordinates(self.mesh['vertices'][triangle])

            grad_test_elem=B_D @ grad_test
      
            v0= J*tf.reduce_sum(w_quad*grad_test_elem[0]*grad_elem)
            v1= J*tf.reduce_sum(w_quad*grad_test_elem[1]*grad_elem)
            v2= J*tf.reduce_sum(w_quad*grad_test_elem[2]*grad_elem)

            l0=J*tf.reduce_sum(w_quad*grad_test_elem[3]*grad_elem)
            l1=J*tf.reduce_sum(w_quad*grad_test_elem[4]*grad_elem)
            l2=J*tf.reduce_sum(w_quad*grad_test_elem[5]*grad_elem)


            indices = tf.constant([[triangle[0]], [triangle[1]], [triangle[2]]])


            
            updates = [v0, v1, v2]
            shape=[self.n_vertices]
            scatter = tf.scatter_nd(indices, updates, shape)
            scatter_=tf.expand_dims(scatter,axis=1)
            
        
            sum_of_vectors = tf.reduce_sum([sum_of_vectors, scatter_], axis=0)


            #a_vertices[triangle[0]]+=a_element[0]

        
            
           
        


        #tf.reduce_sum(a_vertices,axis=1,keepdims=True)-F_total_vertices,tf.reduce_sum(a_edges,axis=1,keepdims=True)-F_total_edges
        return tf.reduce_sum(tf.square(sum_of_vectors-F_total_vertices))


    #@tf.function
    def variational_loss(self):



        a_vertices = self.a_vertices #
        a_edges = self.a_edges #

        n_triangles=self.n_triangles
        xy_quad_total =self.xy_quad_total
        dof=self.dof

        w_quad = tf.concat([self.w_quad.T,self.w_quad.T], axis=0)


        #eval in one shot 
        x_eval=tf.reshape(xy_quad_total,(-1,2))

        grad=self.eval_grad_NN(x_eval)
        grad_=tf.reshape(grad,(n_triangles,-1,2))

        F_total_vertices=self.F_total_vertices
        F_total_edges=self.F_total_edges

   

        grad_test=self.grad_test  




        res=0.0

        for index,triangle in enumerate(self.mesh['triangles']):



            grad_elem=tf.transpose(grad_[index])
   

            # get quadrature points in arb. element and get jacobian
            B,c,J,B_D,B_DD=self.b.change_of_coordinates(self.mesh['vertices'][triangle])

            grad_test_elem=B_D @ grad_test
      
            v0= J*tf.reduce_sum(w_quad*grad_test_elem[0]*grad_elem)
            v1= J*tf.reduce_sum(w_quad*grad_test_elem[1]*grad_elem)
            v2= J*tf.reduce_sum(w_quad*grad_test_elem[2]*grad_elem)

            l0=J*tf.reduce_sum(w_quad*grad_test_elem[3]*grad_elem)
            l1=J*tf.reduce_sum(w_quad*grad_test_elem[4]*grad_elem)
            l2=J*tf.reduce_sum(w_quad*grad_test_elem[5]*grad_elem)



            #a_vertices[triangle[0]]+=a_element[0]
            
            """
            if (self.mesh['vertex_markers'][triangle[0]]==0):
                   # tf.tensor_scatter_nd_update(a_vertices, [triangle[0],1], v0)
                    a_vertices[triangle[0],index].assign(v0)


            if (self.mesh['vertex_markers'][triangle[1]]==0):
                   # tf.tensor_scatter_nd_update(a_vertices, [triangle[1],1], v1)
                    a_vertices[triangle[1],index].assign(v1)
 

            if (self.mesh['vertex_markers'][triangle[2]]==0):
                  #  tf.tensor_scatter_nd_update(a_vertices, [triangle[2],1], v2)
                    a_vertices[triangle[2],index].assign(v2)

            if(self.b.r>=2):
                    if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]==0):
                       # tf.tensor_scatter_nd_update(a_edges, [self.mesh['edges_index_inside_triangle'][index][0],1], l0)
                        a_edges[self.mesh['edges_index_inside_triangle'][index][0],index].assign(l0)

                    if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]==0):
                        #tf.tensor_scatter_nd_update(a_edges, [self.mesh['edges_index_inside_triangle'][index][1],1], l1)
                        a_edges[self.mesh['edges_index_inside_triangle'][index][1],index].assign(l1)

                    if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]==0):
                        #tf.tensor_scatter_nd_update(a_edges, [self.mesh['edges_index_inside_triangle'][index][2],1], l2)
                        a_edges[self.mesh['edges_index_inside_triangle'][index][2],index].assign(l2)

            """

            """
            for r in range(self.b.n):
                a_element[r]= J*tf.reduce_sum(w_quad*grad_test_elem[r]*grad_elem)



            if (self.mesh['vertex_markers'][triangle[0]]==0):
                    a_vertices[triangle[0]]+=a_element[0]


            if (self.mesh['vertex_markers'][triangle[1]]==0):
                    a_vertices[triangle[1]]+=a_element[1]
 

            if (self.mesh['vertex_markers'][triangle[2]]==0):
                    a_vertices[triangle[2]]+=a_element[2]


     
            if(self.b.r>=2):
                    if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][0]]==0):
                        a_edges[self.mesh['edges_index_inside_triangle'][index][0]]+=a_element[3]

                    if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][1]]==0):
                            a_edges[self.mesh['edges_index_inside_triangle'][index][1]]+=a_element[4]

                    if(self.mesh['edge_markers'][self.mesh['edges_index_inside_triangle'][index][2]]==0):
                        a_edges[self.mesh['edges_index_inside_triangle'][index][2]]+=a_element[5]

            #print(F_total_edges)
            """
            break


        #tf.reduce_sum(a_vertices,axis=1,keepdims=True)-F_total_vertices,tf.reduce_sum(a_edges,axis=1,keepdims=True)-F_total_edges
        return tf.reduce_sum(v0+v1+v2+l0+l1+l2)
    




    #@tf.function
    def loss_total(self):
        #loss_b = self.boundary_loss(tape)
        #res = self.variational_loss(tape)
        res=self.custom_loss()
        return  res
    #@tf.function
    def loss_gradient(self):

       # self.a_vertices.assign(tf.zeros((self.n_vertices,self.n_triangles),dtype=tf.float64))
       # self.a_edges.assign(tf.zeros((self.n_edges,self.n_triangles),dtype=tf.float64))
        
        with tf.GradientTape() as tape:
            #loss_grad.watch(self.xy_quad_total)
            tape.watch(self.NN.trainable_variables)
            loss = self.loss_total()
            print(loss)
            #loss=(tf.reduce_sum(tf.square(res_vertices))+tf.reduce_sum(tf.square(res_edges)))/self.dof

        gradient = tape.gradient(loss, self.NN.trainable_variables)
        return loss, gradient

    #@tf.function
    def gradient_descent(self):
        loss, gradient = self.loss_gradient()
        
        self.optimizer.apply_gradients(zip(gradient, self.NN.trainable_variables))
        return loss

    def train(self, iter):



        self.a_vertices = tf.Variable(tf.zeros((self.n_vertices,self.n_triangles),dtype=tf.float64)) #
        self.a_edges = tf.Variable(tf.zeros((self.n_edges,self.n_triangles),dtype=tf.float64)) #

        history = []

        start_time = time.time()
        for i in range(iter+1):

            loss = self.gradient_descent()
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f'Iteration: {i}', f'loss: {loss.numpy():0.10f}', f'time: {elapsed}')

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
        #change that by using interpolator class
        #self.v_evaluations = {}
        #self.v_evaluations["vx_quad"] = self.pb.v(self.x_quad, self.y_quad, self.n_test)
        #self.n_test = np.shape(self.v_evaluations["vx_quad"])[0]
        # print(self.n_test)
        # self.v_evaluations["dv_x_quad"], self.v_evaluations["d2v_x_quad"] = self.pb.dtest_func(self.n_test, self.x_quad)
        # self.v_evaluations["vy_quad"] = self.pb.v_y(self.n_test, self.y_quad)
        # self.v_evaluations["dv_y_quad"], self.v_evaluations["d2v_y_quad"] = self.pb.dtest_func(self.n_test, self.y_quad)
        # print(np.max(self.v_evaluations["v_x_quad"]), np.min(self.v_evaluations["v_x_quad"]), np.sum(self.v_evaluations["v_x_quad"]))
        # print(np.max(self.v_evaluations["v_y_quad"]), np.min(self.v_evaluations["v_y_quad"]), np.sum(self.v_evaluations["v_y_quad"]))


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
        self.w_quad = np.array(weights,dtype=np.float64)
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
        

        for index,triangle in enumerate(self.mesh['triangles']):
            F_element=[]
            x_element=[]
            J_element=[]
            x_quad=self.points

            # get quadrature points in arb. element and get jacobian
            B,c,J,B_D,B_DD=self.b.change_of_coordinates(self.mesh['vertices'][triangle])

            xy_quad_element=(B@x_quad.T +c).T
      
            


            J_element.append(J)

            # evaluate f on arb. quad points
            f_quad_element = self.pb.f_exact(xy_quad_element[:, 0], xy_quad_element[:, 1])

            # do the integral and appnd to total list
            #print([J*np.sum(self.w_quad[:,0]*self.b.Base[:,r]*f_quad_element) for r in range(self.b.n)])
            F_element=[J*np.sum(self.w_quad[:,0]*self.b.Base[:,r]*f_quad_element) for r in range(self.b.n)]

            

            F_element=np.array(F_element,dtype=np.float64)
            J_element=np.array(J_element,dtype=np.float64)

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




    def generate_test_points(self):
        # Test points for plotting really

        lower_bound, upper_bound, _, _ = self.get_domain_info()

        delta_test = self.params['delta_test']
        x_test = np.arange(lower_bound[0], upper_bound[0] + delta_test, delta_test)
        y_test = np.arange(lower_bound[1], upper_bound[1] + delta_test, delta_test)
        data_temp = np.asarray([[[x_test[i], y_test[j], self.pb.u_exact(x_test[i], y_test[j])]
                                 for i in range(len(x_test))] for j in range(len(y_test))])
    
        n_points = len(y_test)

        x_test = data_temp.flatten()[0::3]
        y_test = data_temp.flatten()[1::3]
        exact = data_temp.flatten()[2::3]
        return np.hstack((x_test[:, None], y_test[:, None])), exact[:, None], n_points

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
        

    def plot_domain(self, PLOT):

        a, b, s, _ = self.get_domain_info()

        fontsize = 24
        x_train_plot, y_train_plot = zip(*self.boundary_points)
        fig, ax = plt.subplots(1)
        
        plt.scatter(x_train_plot, y_train_plot, color='red')
        for xc in self.grid_x:
            plt.axvline(x=xc, ymin=0.045, ymax=0.954, linewidth=1.5)
        for yc in self.grid_y:
            plt.axhline(y=yc, xmin=0.045, xmax=0.954, linewidth=1.5)

        plt.xlim([a[0] - 0.05*s[0], b[0] + 0.05*s[0]])
        plt.ylim([a[1] - 0.05*s[1], b[1] + 0.05*s[1]])
        plt.xlabel('$x$', fontsize = fontsize)
        plt.ylabel('$y$', fontsize = fontsize)
        ax.locator_params(nbins=5)
        plt.tick_params( labelsize = 20)
        #fig.tight_layout()
        fig.set_size_inches(w=11,h=11)

        if PLOT == 'save':
            plt.savefig('VPINN_domain.pdf')
        else:
            plt.show()
        

    def plot_prediction(self, PLOT):

        points, _, n_points = self.generate_test_points()

        x = points[:,0:1].flatten()
        y = points[:,1:2].flatten()

        prediction = self.eval_NN(np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)))[0]

        x = np.asarray(np.split(x, n_points))
        y = np.asarray(np.split(y, n_points))
        u = np.reshape(prediction, (n_points, n_points))

        fontsize = 32
        labelsize = 26
        fig_pred, ax_pred = plt.subplots(constrained_layout=True)
        CS_pred = ax_pred.contourf(x, y, u, 100, cmap='jet', origin='lower')
        cbar = fig_pred.colorbar(CS_pred, shrink=0.67)
        cbar.ax.tick_params(labelsize = labelsize)
        ax_pred.locator_params(nbins=8)
        ax_pred.set_xlabel('$x$' , fontsize = fontsize)
        ax_pred.set_ylabel('$y$' , fontsize = fontsize)
        plt.tick_params( labelsize = labelsize)
        ax_pred.set_aspect(1)
        #fig.tight_layout()
        fig_pred.set_size_inches(w=11,h=11)

        if PLOT == 'save':
            plt.savefig('Predict.png')
        else:
            plt.show()



    def plot_exact(self, PLOT):

        points, sol, n_points = self.generate_test_points()

        x = np.asarray(np.split(points[:,0:1].flatten(), n_points))
        y = np.asarray(np.split(points[:,1:2].flatten(), n_points))
        u = np.asarray(np.split(sol.flatten(), n_points))

        fontsize = 32
        labelsize = 26
        fig_ext, ax_ext = plt.subplots(constrained_layout=True)
        CS_ext = ax_ext.contourf(x, y, u, 100, cmap='jet', origin='lower')
        cbar = fig_ext.colorbar(CS_ext, shrink=0.67)
        cbar.ax.tick_params(labelsize = labelsize)
        ax_ext.locator_params(nbins=8)
        ax_ext.set_xlabel('$x$' , fontsize = fontsize)
        ax_ext.set_ylabel('$y$' , fontsize = fontsize)
        plt.tick_params( labelsize = labelsize)
        ax_ext.set_aspect(1)
        #fig.tight_layout()
        fig_ext.set_size_inches(w=11,h=11)

        if PLOT == 'save':
            plt.savefig('Exact.png')
        else:
            plt.show()
        
        
    def plot_pointwise_error(self, PLOT):

        points, sol, n_points = self.generate_test_points()

        x = points[:,0:1].flatten()
        y = points[:,1:2].flatten()

        prediction = self.eval_NN(np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)))[0]
        u_pred = np.reshape(prediction, (n_points, n_points))


        x = np.asarray(np.split(x, n_points))
        y = np.asarray(np.split(y, n_points))
        # u_pred = np.asarray(np.split(prediction, n_points))
    
        u_exact = np.asarray(np.split(sol.flatten(), n_points))

        fontsize = 32
        labelsize = 26
        fig_err, ax_err = plt.subplots(constrained_layout=True)
        CS_err = ax_err.contourf(x, y, abs(u_exact - u_pred), 100, cmap='jet', origin='lower')
        cbar = fig_err.colorbar(CS_err, shrink=0.65, format="%.4f")
        cbar.ax.tick_params(labelsize = labelsize)
        ax_err.locator_params(nbins=8)
        ax_err.set_xlabel('$x$' , fontsize = fontsize)
        ax_err.set_ylabel('$y$' , fontsize = fontsize)
        plt.tick_params( labelsize = labelsize)
        ax_err.set_aspect(1)
        #fig.tight_layout()
        fig_err.set_size_inches(w=11,h=11)

        if PLOT == 'save':
            plt.savefig('Pointwise_Error.png')
        else:
            plt.show()
        