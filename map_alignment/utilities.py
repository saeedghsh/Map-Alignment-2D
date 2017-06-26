'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi (saeed.gh.sh@gmail.com)

This file is part of Arrangement Library.
The of Arrangement Library is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>
'''
from __future__ import print_function

import numpy as np
import scipy.signal

################################################################################
################################################################################
################################################################################
def FindPeaks (signal,
               CWT=False, cwt_range=(1,8,1),
               Refine_win=1 , MinPeakDist = 1 , MinPeakVal=np.spacing(1),
               Polar=False , Polar_range=[-np.pi,np.pi]):
    '''
    1_ cwt_range: finding all the peaks in the signal
    2_ Refine_win: refining the peak position by looking around each peak
    3_ MinPeakDist
       Note that if (Polar = True), the 0=2*np.pi
    4_ MinPeakVal

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    '''

    # 1_ cwt_range
    #    finding all the peaks in the signal
    ### oops actually I'm not using "find_peaks_cwt"!
    ### this function is a peak finder itself!
    ### it considers all points as peaks and then rejects those which are not.
    if CWT:
        peakind = scipy.signal.find_peaks_cwt(signal, np.arange(cwt_range[0],cwt_range[1],cwt_range[2]))
        peakind = list (peakind)
    else:
        peakind = list(np.arange(np.shape(signal)[0]))

    # 2_ Refine_win
    #    looking around each peak to find the maximum in its neighborhood (2*Refine_win)
    #    hence refining the peak position
    for i in np.arange(np.shape(peakind)[0]):
        indices = [j for j in range(max(peakind[i]-Refine_win,0),
                                    min(peakind[i]+Refine_win,len(signal)))]
        neighbors = [signal[j] for j in indices]
        newpeak = max(neighbors)
        newind = neighbors.index(newpeak)
        peakind[i] = indices[newind]

    # 3_ MinPeakDist
    #    cheching consecutive peaks and remove the smaller one if they are too close
    #    starting from the one before last element and coming down to 1st element (inedexing issue in list.pop())
    peakval = list(signal[peakind])
    for i in np.arange(np.shape(peakval)[0]-2,-1,-1):

        # if the coordination system is polar or not
        if Polar:
            dist = polar_distance(peakind[i],  peakind[i+1])
        else:
            dist = peakind[i+1] - peakind[i]

        # check if the consecutive peaks are too close
        if dist < MinPeakDist:
            # if they are too close, pop the smaller one
            if (peakval[i] > peakval[i+1]):
                peakind.pop(i+1)
                peakval.pop(i+1)
            else:
                peakind.pop(i)
                peakval.pop(i)

    # if the coordination system is polar, the first and last elements
    # in the list shall be compared
    if Polar:
        if polar_distance(peakind[0],  peakind[-1]) < MinPeakDist:
            if (peakval[0] < peakval[-1]):
                peakind.pop(0)
            else:
                peakind.pop(-1)           

    # 4_ MinPeakVal
    peakval = list(signal[peakind])
    for i in np.arange(np.shape(peakval)[0]-1,-1,-1):
        if peakval[i] < MinPeakVal * np.max(signal):
            peakind.pop(i)
            peakval.pop(i)

    return peakind

################################################################################
def polar_distance(angle1, angle2, radian=False):
    '''
    '''
    dist = []
    if radian:    
        dist.append( np.abs(angle2 - angle1) )
        dist.append( np.abs(angle2 - (angle1 + (2*np.pi))) )
        dist.append( np.abs(angle2 - (angle1 - (2*np.pi))) )
    else:
        dist.append( np.abs(angle2 - angle1) )
        dist.append( np.abs(angle2 - (angle1 + 360)) )
        dist.append( np.abs(angle2 - (angle1 - 360)) )

    return np.min (dist,0)

def polarDistance(angle1, angle2, radian=True):
    return polar_distance(angle1, angle2, radian)

################################################################################
def Euclidean_distance (P1, P2):
    '''
    '''
    return np.sqrt( np.sum([(P2[i]-P1[i])**2 for i in range(len(P1))]) )

################################################################################
def Gauss1D(x, mu, s):
    '''
    '''
    coefficient = 1.0 / (s*np.sqrt(2*np.pi))
    y = np.exp(- (x-mu)**2 /(2.0 * s**2))
    return coefficient * y

################################################################################
def Gauss2DNormal(Size, Sigma = 0.7, Normalize = '/sum'):
    '''
    '''
    # borrowed from: https://gist.github.com/andrewgiessel 
    # fwhm: full width at half maximum
    # fwhm = 2 * (sqrt(2*(ln(2)))) * sigma

    x = np.arange(0, Size, 1, float)
    y = x[:,np.newaxis]

    x0 = y0 = Size // 2
    
    gauss = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*(Sigma**2)) )
    
    if Normalize == '/sum':
        gauss = gauss / np.sum(gauss)
    elif Normalize == '/2*pi*sigma**2':
        gauss = gauss / (2*np.pi*(Sigma**2))

    return gauss

################################################################################
def GammaFilter (Size=3, Sigma=0.7, Order=1):
    '''
    '''
    GaussKernel = Gauss2DNormal(Size, Sigma, '/2*pi*sigma**2')

    # TODO, Question: which direction is correct for dx?
    # Actually why, since I know which direction!

    dx = np.arange(Size/2, -Size/2, -1, float)
    dx = np.arange(-Size/2, Size/2, 1, float) + 1
    dy = dx[:,np.newaxis]
    
    MN = (-(dx + np.sign(Order)*1j*dy) / Sigma**2)** np.abs(Order)

    return MN * GaussKernel

################################################################################
def OriGradient (Image, KernelSize=7, KernelSigma=1, AngleMode='full', ConvMode='valid', Gamma=.5):
    """
    A function to calculate the "Oriented Gradients" of an input image.
    
    Inputs is a gray-scale image, and the output is an array (np.ndarray). Each value of the output represents the oriented gradient of the corresponding pixel in the input image. The angle and magnitude of the orientated gradient are represented in an Euler format.

    There is difference between the size of input and output, depending on the size of the kernel:
    input.shape[i] = output.shape[i] + (KernelSize-1)
    
    KernelSize and KernelSigma, define the derivative kernel.
    AngleModel controls the angular interval of the output, full=[-pi,pi], half=[-pi/2,pi/2]

    """

    gammafilter = GammaFilter (KernelSize, KernelSigma, Order=1)
    oriented_grad = scipy.signal.convolve2d(Image, gammafilter, mode=ConvMode,
                                            boundary='fill', fillvalue=0)
    
    magnitude = np.abs(oriented_grad)**Gamma

    if (AngleMode == 'full'):
        angle = np.angle(oriented_grad)
    elif (AngleMode == 'half'):
        angle = np.angle( np.exp(2*1j*np.angle(oriented_grad)) ) / 2.0
        
    return magnitude * np.exp(1j * angle)

################################################################################
def wHOG (OrientedGradient, NumBin=180*5, Extension=True):
    '''
    Weighted Histogram of Oriented Gradient, over all of the pixel of the input image.
    It is used for dominant orientation detection.

    The output is a histogram of orientations (angles), where the value of the histogram is coming from the number AND magnitude of all gradients with similar orientations.
    
    "OrientedGradient" from input arguments is the oriented gradient field of an image, not the image itself
    
    "Extension": in order to have continuity at -pi/2 and pi/2 in the histogram of gradients, the signal is extended with value of itself. The portion of extension is 0.2 if (extention=True), otherwise the portion is (extention[0]/extention[1]).    
    '''

    # cutting out the border, because otherwise there will be steep peaks at [0, pi/2., ...]
    # turns out this was not actually true! there is no problem with borders!
    # ori_grad = OrientedGradient[20:-20 , 20:-20]

    ori_grad = OrientedGradient

    h,w = ori_grad.shape[:2]
    Ang_arr = np.reshape(np.angle(ori_grad), h*w, 1)
    Mag_arr = np.reshape(np.abs(ori_grad), h*w, 1)

    hist, bins  = np.histogram(Ang_arr, bins=NumBin, weights=Mag_arr, density=True)
    bincenter = (bins[:-1]+bins[1:])/2

    if Extension:
        if isinstance(Extension, bool):
            a,b = 1,5 # (a/b) = portion to extend at beginning and end
        else:
            a,b = Extension
            
        veclen = hist.shape[0]

        hist1 = hist[ (b-a)*veclen/b:veclen ]
        hist2 = hist
        hist3 = hist[ 0:a*veclen/b ]

        Bin1 = bincenter[ (b-a)*veclen/b:veclen ] - np.pi
        Bin2 = bincenter
        Bin3 = bincenter[ 0:a*veclen/b ] + np.pi

        hist = np.concatenate([hist1,hist2,hist3])
        bincenter = np.concatenate([Bin1,Bin2,Bin3])

    return hist, bincenter

################################################################################
def smooth(x, window_len=11, window='hanning'):
    """
    smooth the data using a window with requested size.
    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1: raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:  raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:  return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    # saesha modify
    ds = y.shape[0] - x.shape[0] # difference of shape
    dsb = ds//2 # [almsot] half of the difference of shape for indexing at the begining
    dse = ds - dsb # rest of the difference of shape for indexing at the end
    y = y[dsb:-dse]

    return y


