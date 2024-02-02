from  my_types import *

print('settings_lib imported ')


class PROBDEF:

    def __init__(self):
        print()
        
    def u_exact(self, x, y):
        utemp = tf.cos(2*np.pi*x)*tf.sin(2*np.pi*y)
        return utemp

    def f_exact(self, x, y):
        gtemp =4*np.pi*np.pi*(tf.cos(2*np.pi*x)*tf.sin(2*np.pi*y)+tf.sin(2*np.pi*y)*tf.cos(2*np.pi*x))
        return gtemp

   