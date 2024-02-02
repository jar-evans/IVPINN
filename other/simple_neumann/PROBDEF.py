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
        utemp = tf.exp(-x**2 -y**2)
        return utemp
    
    def u_exact_np(self, x, y):
        """
        Exact solution with numpy
        """
        utemp = np.exp(-x**2 -y**2)
        return utemp
    
    def mu(self, x, y):
        """
        Mu function
        """
        return (2 + np.sin(x + 2*y))*0 + 1
    
    def beta(self, x, y):
        """
        Beta functions
        """
        beta1=np.sqrt(x-(y**2) +5) 
        beta2=np.sqrt(y-(x**2) +5)
        return beta1*0, beta2*0
    
    def sigma(self, x, y):
        """
        Sigma function
        """
        return (np.exp((x/2) -(y/3)) +2)*0
        
    def neumann(self, x, y): 
        """
        Neumann conditions
        Must be equal to the solution you want to impose on the particular edge
        """   
        f = lambda x: 0*x
        g = lambda x: -2*np.exp(-x*x - 1)
        return - f(x)*(1-y) + (y)*g(x)
    
    def dirichlet(self, x, y):
        """
        Dirichlet conditions
        Must be equal to the solution you want to impose on the particular edge
        """   
        ## For Dirichlet on boundaries 2 and 4
        return (1-x)*(tf.exp(-y**2)) + (x)*(tf.exp(-y**2 - 1))
    
        ## For Dirichlet on all boundaries
        # expre =  tf.sin(3.2*x*(x - y))*tf.cos(x + 4.3*y) + tf.sin(4.6*x + 9.2*y)*tf.cos(5.2*x - 2.6*y)
        # return expre * (10*tf.sin(x*(x-1)*(y)*(y-1)) + 1)
    
    def f(self, x, y):
        """
        Forcing term of the problem
        """
        expression = 4*(1-x*x-y*y)*tf.exp(-x*x -y*y)
        return expression
    
    def dudx(self, x, y):
        return -2*x*np.exp(-x*x -y*y)
    def dudy(self, x, y):
        return -2*y*np.exp(-x*x -y*y)

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
    
