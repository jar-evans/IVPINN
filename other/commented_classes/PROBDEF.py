from  my_types import *

print('settings_lib imported ')
print()

class PROBDEF:

    def __init__(self):
        print()

    def u_exact(self, x, y):
        """
        Exact solution with tensorflow
        """
        utemp = tf.sin(3.2*x*(x - y))*tf.cos(x + 4.3*y) + tf.sin(4.6*x + 9.2*y)*tf.cos(5.2*x - 2.6*y)
        return utemp
    
    def u_exact_np(self, x, y):
        """
        Exact solution with numpy
        """
        utemp = np.sin(3.2*x*(x - y))*np.cos(x + 4.3*y) + np.sin(4.6*x + 9.2*y)*np.cos(5.2*x - 2.6*y)
        return utemp
    
    def mu(self, x, y):
        """
        Mu function
        """
        return 2 + np.sin(x + 2*y)
    
    def beta(self, x, y):
        """
        Beta functions
        """
        beta1=np.sqrt(x-(y**2) +5) 
        beta2=np.sqrt(y-(x**2) +5)
        return beta1, beta2
    
    def sigma(self, x, y):
        """
        Sigma function
        """
        return np.exp((x/2) -(y/3)) +2
        
    def neumann(self, x, y): 
        """
        Neumann conditions
        Must be equal to the solution you want to impose on the particular edge
        """   
        f = lambda x: (np.sin(x) + 2)*(-3.2*x*np.cos(x)*np.cos(3.2*x**2) - 4.3*np.sin(x)*np.sin(3.2*x**2) + 2.6*np.sin(4.6*x)*np.sin(5.2*x) + 9.2*np.cos(4.6*x)*np.cos(5.2*x))
        g = lambda x: (np.sin(x + 2) + 2)*(-3.2*x*np.cos(3.2*x*(x - 1))*np.cos(x + 4.3) - 4.3*np.sin(3.2*x*(x - 1))*np.sin(x + 4.3) + 2.6*np.sin(4.6*x + 9.2)*np.sin(5.2*x - 2.6) + 9.2*np.cos(4.6*x + 9.2)*np.cos(5.2*x - 2.6))
        return - f(x)*(1-y) + (y)*g(x)
    
    def dirichlet(self, x, y):
        """
        Dirichlet conditions
        Must be equal to the solution you want to impose on the particular edge
        """   
        ## For Dirichlet on boundaries 2 and 4
        # return (1-x)*(tf.sin(9.2*y)*tf.cos(2.6*y)) + (x)*(-tf.sin(3.2*y - 3.2)*tf.cos(4.3*y + 1) + tf.sin(9.2*y + 4.6)*tf.cos(2.6*y - 5.2))
    
        ## For Dirichlet on all boundaries
        expre =  tf.sin(3.2*x*(x - y))*tf.cos(x + 4.3*y) + tf.sin(4.6*x + 9.2*y)*tf.cos(5.2*x - 2.6*y)
        return expre * (10*tf.sin(x*(x-1)*(y)*(y-1)) + 1)
    
    def f(self, x, y):
        """
        Forcing term of the problem
        """
        expression = (np.sin(3.2*x*(x - y))*np.cos(x + 4.3*y) + np.sin(4.6*x + 9.2*y)*np.cos(5.2*x - 2.6*y))*(np.exp(x/2 - y/3) + 2) - (np.sin(x + 2*y) + 2)*(-10.24*(x**2)*np.sin(3.2*x*(x - y))*np.cos(x + 4.3*y) + 27.52*x*np.sin(x + 4.3*y)*np.cos(3.2*x*(x - y)) - 18.49*np.sin(3.2*x*(x - y))*np.cos(x + 4.3*y) - 91.4*np.sin(4.6*x + 9.2*y)*np.cos(5.2*x - 2.6*y) + 47.84*np.sin(5.2*x - 2.6*y)*np.cos(4.6*x + 9.2*y)) - (np.sin(x + 2*y) + 2)*(-40.96*((x - 0.5*y)**2)*np.sin(3.2*x*(x - y))*np.cos(x + 4.3*y) - 2*(6.4*x - 3.2*y)*np.sin(x + 4.3*y)*np.cos(3.2*x*(x - y)) - np.sin(3.2*x*(x - y))*np.cos(x + 4.3*y) - 48.2*np.sin(4.6*x + 9.2*y)*np.cos(5.2*x - 2.6*y) - 47.84*np.sin(5.2*x - 2.6*y)*np.cos(4.6*x + 9.2*y) + 6.4*np.cos(3.2*x*(x - y))*np.cos(x + 4.3*y)) + np.sqrt(x - (y**2) + 5)*((6.4*x - 3.2*y)*np.cos(3.2*x*(x - y))*np.cos(x + 4.3*y) - np.sin(3.2*x*(x - y))*np.sin(x + 4.3*y) - 5.2*np.sin(4.6*x + 9.2*y)*np.sin(5.2*x - 2.6*y) + 4.6*np.cos(4.6*x + 9.2*y)*np.cos(5.2*x - 2.6*y)) + np.sqrt((-x**2) + y + 5)*(-3.2*x*np.cos(3.2*x*(x - y))*np.cos(x + 4.3*y) - 4.3*np.sin(3.2*x*(x - y))*np.sin(x + 4.3*y) + 2.6*np.sin(4.6*x + 9.2*y)*np.sin(5.2*x - 2.6*y) + 9.2*np.cos(4.6*x + 9.2*y)*np.cos(5.2*x - 2.6*y)) - 2*(-3.2*x*np.cos(3.2*x*(x - y))*np.cos(x + 4.3*y) - 4.3*np.sin(3.2*x*(x - y))*np.sin(x + 4.3*y) + 2.6*np.sin(4.6*x + 9.2*y)*np.sin(5.2*x - 2.6*y) + 9.2*np.cos(4.6*x + 9.2*y)*np.cos(5.2*x - 2.6*y))*np.cos(x + 2*y) - ((6.4*x - 3.2*y)*np.cos(3.2*x*(x - y))*np.cos(x + 4.3*y) - np.sin(3.2*x*(x - y))*np.sin(x + 4.3*y) - 5.2*np.sin(4.6*x + 9.2*y)*np.sin(5.2*x - 2.6*y) + 4.6*np.cos(4.6*x + 9.2*y)*np.cos(5.2*x - 2.6*y))*np.cos(x + 2*y)
        return expression
    
    def dudx(self, x, y):
        return (6.4*x - 3.2*y)*np.cos(3.2*x*(x - y))*np.cos(x + 4.3*y) - np.sin(3.2*x*(x - y))*np.sin(x + 4.3*y) - 5.2*np.sin(4.6*x + 9.2*y)*np.sin(5.2*x - 2.6*y) + 4.6*np.cos(4.6*x + 9.2*y)*np.cos(5.2*x - 2.6*y)
    def dudy(self, x, y):
        return -3.2*x*np.cos(3.2*x*(x - y))*np.cos(x + 4.3*y) - 4.3*np.sin(3.2*x*(x - y))*np.sin(x + 4.3*y) + 2.6*np.sin(4.6*x + 9.2*y)*np.sin(5.2*x - 2.6*y) + 9.2*np.cos(4.6*x + 9.2*y)*np.cos(5.2*x - 2.6*y)

    # def generate_points_on_edge(self,point1, point2, num_points):
    #     x_vals = np.linspace(point1[0], point2[0], num_points,endpoint=False)
    #     y_vals = np.linspace(point1[1], point2[1], num_points,endpoint=False)
    #     points_on_edge = np.column_stack((x_vals, y_vals))
    #     return points_on_edge

    # def generate_rectangle_points(self,a, b, c, d, num_points_per_edge,bool):

    #     if bool==True:
    #         edge1 = self.generate_points_on_edge(a, b, num_points_per_edge)
    #         edge2 = self.generate_points_on_edge(b, c, num_points_per_edge)
    #         edge3 = self.generate_points_on_edge(c, d, num_points_per_edge)
    #         edge4 = self.generate_points_on_edge(d, a, num_points_per_edge)
    #     else:
    #         cheb_points=(np.cos(np.linspace(0, np.pi, num_points_per_edge + 2)[0:])+1.0)/2
    #         cheb_points=cheb_points[::-1]

    #         x=cheb_points[:-1]
    #         y=0.0*x

    #         edge1=np.transpose(np.vstack((x,y)))

    #         y=cheb_points[:-1]
    #         x=0.0*y +1.0

    #         edge2=np.transpose(np.vstack((x,y)))

    #         x=cheb_points[::-1] [:-1]
    #         y=0.0*x +1.0

    #         edge3=np.transpose(np.vstack((x,y)))


    #         y=cheb_points[::-1] [:-1]
    #         x=0.0*y 

    #         edge4=np.transpose(np.vstack((x,y)))



    #     rectangle_points = np.vstack((edge1, edge2, edge3, edge4))
    #     return rectangle_points  
    

    # def generate_boundary_points(self,n,bool):
    # # Boundary points
    #     a = (0, 0)
    #     b = (1, 0)
    #     c = (1, 1)
    #     d = (0, 1)


    #     boundary_points = self.generate_rectangle_points(a, b, c, d, n,bool)

    #     boundary_points=tf.constant(boundary_points,dtype=tf.float64)


    #     return boundary_points
    
