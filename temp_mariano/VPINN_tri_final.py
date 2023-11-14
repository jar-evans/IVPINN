import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights
import time
SEED = 42


class VPINN():

    def __init__(self, pb, params, mesh, NN = None):

        super().__init__()

        # accept parameters
        self.pb = pb
        self.params = params
        self.NN_struct = params['NN_struct']
        self.n_test = params['n_test']

        self.mesh = mesh

        self.n_el_x, self.n_el_y = self.params['n_elements']

        # generate all points/coordinates to be used in the process
        self.generate_boundary_points()
        # self.generate_inner_points()
        self.pre_compute()

        # add the neural network to the class if given at initialisation
        if NN:
            self.set_NN(NN)

    def set_NN(self, NN, LR):
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        # initialise the NN
        self.NN = NN

        # take trainable vars
        self.vars = self.NN.trainable_variables

        # set optimiser
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    def initialise_NN(self, structure: list, LR: float):

        input_dim = structure[0]
        output_dim = structure[-1]
        network = structure[1:-1]

        NN = tf.keras.Sequential()
        NN.add(tf.keras.layers.InputLayer(input_dim))
        # NN.add(tf.keras.layers.Lambda(lambda x: 2. * (x + 1) / (2) - 1.))

        for width in network:
            NN.add(tf.keras.layers.Dense(width,
                                            activation='tanh',
                                            use_bias=True,
                                            kernel_initializer='glorot_normal',
                                            bias_initializer='zeros'))
        NN.add(tf.keras.layers.Dense(output_dim))

        self.set_NN(NN, LR)

    def eval_NN(self, x, y):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        y = tf.convert_to_tensor(y, dtype=tf.float64)        

        with tf.GradientTape(persistent=True) as second_order:
            second_order.watch(x)
            second_order.watch(y)
            with tf.GradientTape(persistent=True) as first_order:
                first_order.watch(x)
                first_order.watch(y)
                u = self.NN(tf.concat([x, y], 1))
            d1xu = first_order.gradient(u, x)
            d1yu = first_order.gradient(u, y)
        d2xu = second_order.gradient(d1xu, x)
        d2yu = second_order.gradient(d1yu, y)

        del first_order
        del second_order

        return u, [d1xu, d1yu], [d2xu, d2yu]
    
    def u_NN(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        y = tf.convert_to_tensor(y, dtype=tf.float64)  
        return self.NN(tf.concat([x, y], 1))

    def boundary_loss(self):
    ## NOTE:impose boundary or same structure for ICs
        boundary_x = self.boundary_points[:,0].flatten()
        boundary_y = self.boundary_points[:,1].flatten()

        prediction = self.eval_NN(np.reshape(boundary_x, (len(boundary_x), 1)), np.reshape(boundary_y, (len(boundary_y), 1)))[0]

        u_bound_NN = prediction
        u_bound_exact = self.pb.u_exact(boundary_x, boundary_y)

        return tf.reduce_mean(tf.square(u_bound_NN - u_bound_exact))

    def variational_loss(self):

        v_x = self.v_evaluations["vx_quad"]
        # v_y = self.v_evaluations["vy_quad"]
        # dv_x_quad_el, d2v_x_quad_el = self.v_evaluations["dv_x_quad"], self.v_evaluations["d2v_x_quad"]
        # v_y_quad_el = self.v_evaluations["v_y_quad"]
        # dv_y_quad_el, d2v_y_quad_el = self.v_evaluations["dv_y_quad"], self.v_evaluations["d2v_y_quad"]

        varloss_total = 0

        for element in range(self.mesh.N):

            F_element = self.F_total[element]
            xy_quad_element = self.xy_quad_total[element]
            J = self.J_total[element]

            # print(np.sum(self.F_total))
            # print(np.max(self.F_total))
            # print(np.min(self.F_total))
            # break

            x_quad_element = np.reshape(xy_quad_element[:, 0], (len(xy_quad_element),1))
            y_quad_element = np.reshape(xy_quad_element[:, 1], (len(xy_quad_element),1))

            u_NN_quad_el, [d1xu_NN_quad_el, d1yu_NN_quad_el], [d2xu_NN_quad_el, d2yu_NN_quad_el] = self.eval_NN(x_quad_element, y_quad_element)
            integrand_1 = d2xu_NN_quad_el + d2yu_NN_quad_el

            # print(np.shape(self.w_quad))
            # print(np.shape(v_x_quad_el[1]))
            # print(np.shape(self.w_quad*v_x_quad_el[1]*v_y_quad_el[1]*integrand_1))


            if self.params['var_form'] == 0:
                u_NN_el = tf.convert_to_tensor([J*tf.reduce_sum(self.w_quad*v_x[r]*integrand_1) for r in range(self.n_test)], dtype=tf.float64)

            # if self.params['var_form'] == 1:
            #     u_NN_el_1 = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(
            #         self.w_quad[:, 0:1]*dv_x_quad_el[r]*self.w_quad[:, 1:2]*v_y_quad_el[k]*d1xu_NN_quad_el)
            #         for r in range(n_test_x)] for k in range(n_test_y)], dtype=tf.float32)
            #     u_NN_el_2 = tf.convert_to_tensor([[jacobian/jacobian_y*tf.reduce_sum(
            #         self.w_quad[:, 0:1]*v_x_quad_el[r]*self.w_quad[:, 1:2]*dv_y_quad_el[k]*d1yu_NN_quad_el)
            #         for r in range(n_test_x)] for k in range(n_test_y)], dtype=tf.float32)
            #     u_NN_el = - u_NN_el_1 - u_NN_el_2

            # if self.params['var_form'] == 2:
            #     u_NN_el_1 = tf.convert_to_tensor([[jacobian*tf.reduce_sum(
            #         self.w_quad[:, 0:1]*d2v_x_quad_el[r]*self.w_quad[:, 1:2]*v_y_quad_el[k]*u_NN_quad_el)
            #         for r in range(n_test_x)] for k in range(n_test_y)], dtype=tf.float32)
            #     u_NN_el_2 = tf.convert_to_tensor([[jacobian*tf.reduce_sum(
            #         self.w_quad[:, 0:1]*v_x_quad_el[r]*self.w_quad[:, 1:2]*d2v_y_quad_el[k]*u_NN_quad_el)
            #         for r in range(n_test_x)] for k in range(n_test_y)], dtype=tf.float32)
            #     u_NN_el = u_NN_el_1 + u_NN_el_2

            res_NN_element = tf.reshape(u_NN_el - F_element, [1, -1])
            loss_element = tf.reduce_mean(tf.square(res_NN_element))
            varloss_total = varloss_total + loss_element

        return varloss_total

    @tf.function
    def loss_total(self):
        loss_0 = 0
        loss_b = self.boundary_loss()
        loss_v = self.variational_loss()
        return tf.cast(loss_0, dtype=tf.float64) + tf.cast(loss_b, dtype=tf.float64) + tf.cast(loss_v, dtype=tf.float64)

    @tf.function
    def loss_gradient(self):
        with tf.GradientTape(persistent=True) as loss_grad:
            loss = self.loss_total()
        gradient = loss_grad.gradient(loss, self.vars)
        return loss, gradient

    @tf.function
    def gradient_descent(self):
        loss, gradient = self.loss_gradient()
        self.optimizer.apply_gradients(zip(gradient, self.vars))
        return loss

    def train(self, iter):

        history = []

        start_time = time.time()
        for i in range(iter):

            loss = self.gradient_descent()

            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f'Iteration: {i}', f'loss: {loss.numpy():0.6f}', f'time: {elapsed}')
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
        a, b, scale, mid = self.get_domain_info()

        boundary_grid = scale*(lhs(1, self.params['n_bound']) - 0.5) + mid

        boundary_grid = np.array(boundary_grid)

        y_up = b[1]*np.ones((len(boundary_grid[:, 0]), 1))
        y_lo = a[1]*np.ones((len(boundary_grid[:, 0]), 1))
        x_ri = b[0]*np.ones((len(boundary_grid[:, 1]), 1))
        x_le = a[0]*np.ones((len(boundary_grid[:, 1]), 1))

        u_up_train = self.pb.u_exact(boundary_grid[:, 0], y_up)
        x_up_train = np.hstack(
            (np.reshape(boundary_grid[:, 0], (len(boundary_grid[:, 0]), 1)), y_up))

        u_lo_train = self.pb.u_exact(boundary_grid[:, 0], y_lo)
        x_lo_train = np.hstack(
            (np.reshape(boundary_grid[:, 0], (len(boundary_grid[:, 0]), 1)), y_lo))

        u_ri_train = self.pb.u_exact(boundary_grid[:, 1], x_ri)
        x_ri_train = np.hstack(
            (x_ri, np.reshape(boundary_grid[:, 1], (len(boundary_grid[:, 1]), 1))))

        u_le_train = self.pb.u_exact(boundary_grid[:, 1], x_le)
        x_le_train = np.hstack(
            (x_le, np.reshape(boundary_grid[:, 1], (len(boundary_grid[:, 1]), 1))))

        self.boundary_points = np.concatenate(
            (x_up_train, x_lo_train, x_ri_train, x_le_train))
        self.boundary_sol = np.concatenate(
            (u_up_train, u_lo_train, u_ri_train, u_le_train))

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


        self.b=interpolator(self.n_test,False,True,points=self.points)

        self.B=interpolator(self.n_inter,False,False,points=None)

    def generate_quadrature_points(self):
        """
        here you will have col vectors with correct stuff defined on the ref triangle
        self.x_quad
        self.y_quad 
        self.w_quad
        """
        self.xy_quad, self.w_quad = self.mesh.GLQ()
        self.x_quad = np.reshape(self.xy_quad[:,0], (len(self.xy_quad), 1))
        self.y_quad = np.reshape(self.xy_quad[:,1], (len(self.xy_quad), 1))


        self.points=self.xy_quad
        
        self.w_quad = np.reshape(self.w_quad, (len(self.w_quad), 1))

    def pre_compute(self):
        self.generate_quadrature_points()
        self.evaluate_test_functions()
        self.F_ext_total = self.construct_RHS()

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

        F_total = []
        xy_quad_total = []
        J_total = []

        for big_element in range(self.mesh.N):
            F_element=[]
            x_element=[]
            J_element=[]

            for element in range(self.mesh.meshed_elements[big_element].N):
    

                # get quadrature points in arb. element and get jacobian
                xy_quad_element, J = self.mesh.translate(self.xy_quad, self.mesh.meshed_elements[big_element]._get_element_points(element))
                x_element.append(xy_quad_element)
                J_element.append(J)

                # evaluate f on arb. quad points
                f_quad_element = self.pb.f_exact(xy_quad_element[:, 0], xy_quad_element[:, 1])

                # do the integral and appnd to total list
                F_element.append([J*np.sum(self.w_quad[:,0]*self.b.Base[:,r]*f_quad_element) for r in range(self.b.n)])

            

            F_element=np.array(F_element,dtype=np.float64)
            print(F_element)
            J_element=np.array(J_element,dtype=np.float64)
            xy_quad_total.append(x_element)
            J_total.append(J_element)
            F_total.append(F_element)

            #xy_quad_total.append(xy_quad_element)
            #J_total.append(J)

        self.J_total=J_total
        self.F_total = F_total
        self.xy_quad_total = np.array(xy_quad_total,dtype=np.float64)





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
        