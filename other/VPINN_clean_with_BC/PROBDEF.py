from  my_types import *

print('settings_lib imported ')
print()


class PROBDEF:

    def __init__(self):
        print()

    def u_exact(self, x, y):
        utemp = tf.cos(np.pi*(x+0.5))*tf.sin(np.pi*y)
        return utemp

    def f_exact(self, x, y):
        gtemp =2*np.pi*np.pi*(tf.cos(np.pi*(x+0.5))*tf.sin(np.pi*y))
        return gtemp


