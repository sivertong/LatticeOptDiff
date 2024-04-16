import scipy.io as io
import numpy as np
#
# io.savemat('testDataMAT.mat', mdict=np.load('D:\CodeSave\GitCode\\topodiff\scripts\generated\samples_200x11x1.npz'))


# io.savemat('DenoiseVideo.mat', mdict=np.load('D:\CodeSave\GitCode\\topodiff\scripts\DenoiseVideo.npz'))

io.savemat('D:\CodeSave\GitCode\IHDiff\scripts\samples_600x11x1.mat', mdict=np.load("D:\CodeSave\GitCode\IHDiff\scripts\samples_600x11x1.npz"))