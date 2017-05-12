import sys
import cv2
import numpy as np
import scipy

import sys
sys.path.append( u'../core/' )
sys.path.append( u'/home/saesha/Dropbox/myGits/arrangement/')
from map_alignment import skiz_bitmap
########################################
# http://scipy.github.io/devdocs/generated/scipy.spatial.Voronoi.html
# https://pythonhosted.org/pymorph/
# http://www.inf.u-szeged.hu/~palagyi/skel/skel.html



def main( file_name, save, vis ):
    ''''''
    try:
        # no flipup, because won't flip in saving and origin uppper in visualization
        image = cv2.imread( file_name, cv2.IMREAD_GRAYSCALE)
    except:
        raise ( NameError('invalid file name') )    
    
    skiz, dis = skiz_bitmap(image, invert=True, return_distance=True )

    if vis:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,3, figsize=(20,12))
        axes[0].imshow(image, cmap='gray', interpolation='nearest')
        axes[1].imshow(dis, cmap='gray', interpolation='nearest')
        axes[2].imshow(skiz, cmap='gray', interpolation='nearest')
        plt.tight_layout()
        plt.show()

    if save:
        file_name_skiz =  file_name[:-4]+ '_skiz.png'
        file_name_dis =  file_name[:-4]+ '_dis.png'
        # scipy.misc.imsave(dir_adr+file_name_png, ogm) # this normalizes the image
        scipy.misc.toimage(skiz, cmin=0, cmax=skiz.max()).save(file_name_skiz)
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

    save = False
    vis = False
    file_name = None

    args = sys.argv

    if '--h' in args[1:]:
        print (__main__.__doc__)
    
    for arg in args[1:]:
        if '--s' in arg:
            save = True
        elif '--v' in arg:
            vis = True
        elif '.png' in arg:
            file_name = arg

    # if file name is not provided, set to default
    if file_name is None:
        raise (NameError('no png file is found'))

    # if option is not specified, default is set to '--v'
    if not(save) and not(vis):
        vis =True


    main( file_name, save, vis )
    
    
