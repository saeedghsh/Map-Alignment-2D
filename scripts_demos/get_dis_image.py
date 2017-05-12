import sys
import cv2
import numpy as np
import scipy.ndimage

########################################
# http://scipy.github.io/devdocs/generated/scipy.spatial.Voronoi.html
# https://pythonhosted.org/pymorph/
# http://www.inf.u-szeged.hu/~palagyi/skel/skel.html

def main( file_name ):
    ''''''
    try:
        # no flipup, because won't flip in saving and origin uppper in visualization
        image = cv2.imread( file_name, cv2.IMREAD_GRAYSCALE)
    except:
        raise ( NameError('invalid file name') )    

    img_binary = np.where( image < 128, 0, 255 )
    dis = scipy.ndimage.morphology.distance_transform_bf( img_binary )
    # scipy.ndimage.morphology.distance_transform_bf - Distance transform function by a brute force algorithm.
    # scipy.ndimage.morphology.distance_transform_edt - Exact euclidean distance transform.
    # scipy.ndimage.morphology.distance_transform_cdt - Distance transform for chamfer type of transforms.

    
    file_name_dis = file_name[:-4]+ '_dis2.png'
    scipy.misc.toimage(dis, cmin=0, cmax=dis.max()).save(file_name_dis)


if __name__ == '__main__':
    '''
    image_name = '../place_categorization_2D/map_sample/kpt4a_full.png'
    mage_name = '20170131163311_.png'
    image_name = '20170131135829_edit.png'
    image_name = '/home/saesha/Dropbox/myGits/sample_data/HH/HIH_03.png'
    image_name = '/home/saesha/Documents/tango/E5_5/20170205104625.png'

    example:
    python skiz_demo.py --s --v filename

    '''

    file_name = None
    args = sys.argv
    
    for arg in args[1:]:
        if '.png' in arg: file_name = arg

    # if file name is not provided, set to default
    if file_name is None:
        raise (NameError('no png file is found'))

    main( file_name )
    
    
