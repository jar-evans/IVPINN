from  my_types import *

print('settings_lib imported ')
print()


class PROBDEF:

    def __init__(self):
        print()

    def u_exact(self, x, y):
        utemp = tf.tanh(2*(tf.pow(x,3) -tf.pow(y,4)))
        return utemp
    
    def u_exact_np(self, x, y):
        utemp = np.tanh(2*(np.power(x,3) -np.power(y,4)))
        return utemp

    def f_exact(self, x, y):
        gtemp = +4*(1/tf.pow(tf.math.cosh(2*(tf.pow(x,3) - tf.pow(y,4))),2))*(-3*x + 6*tf.pow(y,2) +2*(9*tf.pow(x,4) + 16*tf.pow(y,6))*tf.tanh(2*(tf.pow(x,3) - tf.pow(y,4))))
        return gtemp



    def generate_points_on_edge(self,point1, point2, num_points):
        x_vals = np.linspace(point1[0], point2[0], num_points,endpoint=False)
        y_vals = np.linspace(point1[1], point2[1], num_points,endpoint=False)
        points_on_edge = np.column_stack((x_vals, y_vals))
        return points_on_edge
    
    def chebicev(self,num_points_per_edge):
        pass



    def generate_rectangle_points(self,a, b, c, d, num_points_per_edge,bool):

        if bool==True:
            edge1 = self.generate_points_on_edge(a, b, num_points_per_edge)
            edge2 = self.generate_points_on_edge(b, c, num_points_per_edge)
            edge3 = self.generate_points_on_edge(c, d, num_points_per_edge)
            edge4 = self.generate_points_on_edge(d, a, num_points_per_edge)
        else:
            cheb_points=(np.cos(np.linspace(0, np.pi, num_points_per_edge + 2)[0:])+1.0)/2
            cheb_points=cheb_points[::-1]

            x=cheb_points[:-1]
            y=0.0*x

            edge1=np.transpose(np.vstack((x,y)))

            y=cheb_points[:-1]
            x=0.0*y +1.0

            edge2=np.transpose(np.vstack((x,y)))

            x=cheb_points[::-1] [:-1]
            y=0.0*x +1.0

            edge3=np.transpose(np.vstack((x,y)))


            y=cheb_points[::-1] [:-1]
            x=0.0*y 

            edge4=np.transpose(np.vstack((x,y)))



        rectangle_points = np.vstack((edge1, edge2, edge3, edge4))
        return rectangle_points  
    

    def generate_boundary_points(self,n,bool):
    # Boundary points
        a = (0, 0)
        b = (1, 0)
        c = (1, 1)
        d = (0, 1)


        boundary_points = self.generate_rectangle_points(a, b, c, d, n,bool)

        boundary_points=tf.constant(boundary_points,dtype=tf.float64)


        return boundary_points