import sys
new_paths = [
    u'../Python-CPD/',
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

### for Python-CPD
from functools import partial
from core import (RigidRegistration, AffineRegistration)

################################################################################
def visualize(X, Y, ax):
    '''
    This method is for the animation of CPD from:
    https://github.com/siavashk/Python-CPD

    X: destination
    Y: source
    '''

    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red')
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue')
    ax.axis('equal')
    plt.draw()
    plt.pause(.01**5)

################################################################################
src = np.load('examples/aligne_optimize/CPD_E5_05_tango/src_point_cloud.npy')
dst = np.load('examples/aligne_optimize/CPD_E5_05_tango/dst_point_cloud.npy')
R = np.load('examples/aligne_optimize/CPD_E5_05_tango/rot.npy')
t = np.load('examples/aligne_optimize/CPD_E5_05_tango/tra.npy')
s = np.load('examples/aligne_optimize/CPD_E5_05_tango/sca.npy')


### registration
fig = plt.figure()
fig.add_axes([0, 0, 1, 1])
callback = partial(visualize, ax=fig.axes[0])

reg = RigidRegistration( dst,src,
                         R=R, t=t, s=s,
                         sigma2=None, maxIterations=100, tolerance=0.001)

# reg = AffineRegistration( dst, src,
#                           B= s*R, t=t,
#                           sigma2=None, maxIterations=100, tolerance=0.001)

Y_transformed, s, R, t = reg.register(callback)
plt.show()
print (reg.err)

fig = plt.figure()
fig.add_axes([0, 0, 1, 1])
visualize(reg.X, reg.Y, fig.axes[0])
plt.show()
