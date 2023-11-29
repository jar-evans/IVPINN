from  my_types import *

print('settings_lib imported ')
print()

class PROBDEF:

    def __init__(self):
        print()

    def u_exact(self, x, y):
        
        utemp = tf.cos(np.pi*x)*tf.exp(-y)
        return utemp

    def f_exact(self, x, y):

        """
        A = np.pi*np.pi*(tf.sin(np.pi*x))
        A *= tf.sin(tf.sin(np.pi*x))
        A *= tf.exp(-y)

        B = -np.pi*np.pi*(tf.cos(np.pi*x))**2
        B *= tf.cos(tf.sin(np.pi*x))*tf.exp(-y)


        gtemp =     -A -B -tf.cos(tf.sin(np.pi*x))*tf.exp(-y)


        gtemp= -tf.cos(tf.sin(np.pi*x))*tf.exp(-y)- np.pi*np.pi*(tf.sin(np.pi*x))*tf.sin(tf.sin(np.pi*x))+np.pi*np.pi*(tf.cos(np.pi*x))**2 *tf.cos(tf.sin(np.pi*x))*tf.exp(-y)
        """
        gtemp=-tf.exp(-y)*tf.cos(np.pi*x)  +np.pi*np.pi*tf.cos(np.pi*x)*tf.exp(-y)
        return gtemp



