#!/bin/bash -ue

file_list=(
    '/home/saesha/Documents/tango/kpt4a_f/20170131163311.png'
    # '/home/saesha/Documents/tango/kpt4a_kb/20170131163634.png'
    # '/home/saesha/Documents/tango/kpt4a_kl/20170131162628.png'
    # '/home/saesha/Documents/tango/kpt4a_lb/20170131164048.png'
    
    # '/home/saesha/Documents/tango/HIH_01_full/20170131135829.png'
    # '/home/saesha/Documents/tango/HIH_02/20170409123351.png'
    # '/home/saesha/Documents/tango/HIH_03/20170409123544.png'
    # '/home/saesha/Documents/tango/HIH_04/20170409123754.png'
    
    # '/home/saesha/Documents/tango/E5_1/20170131150415.png'
    # '/home/saesha/Documents/tango/E5_2/20170131131405.png'
    # '/home/saesha/Documents/tango/E5_3/20170131130616.png'
    # '/home/saesha/Documents/tango/E5_4/20170131122040.png'
    # '/home/saesha/Documents/tango/E5_5/20170205104625.png'
    # '/home/saesha/Documents/tango/E5_6/20170205105917.png'
    # '/home/saesha/Documents/tango/E5_7/20170205111301.png'
    # '/home/saesha/Documents/tango/E5_8/20170205112339.png'
    # '/home/saesha/Documents/tango/E5_9/20170205110552.png'
    # '/home/saesha/Documents/tango/E5_10/20170205111807.png'
    # '/home/saesha/Documents/tango/E5_11/20170409125554.png'
    # '/home/saesha/Documents/tango/E5_12/20170409130127.png'
    # '/home/saesha/Documents/tango/E5_13/20170409130542.png'
    # '/home/saesha/Documents/tango/E5_14/20170409131152.png'

    # '/home/saesha/Documents/tango/F5_1/20170131132256.png'
    # '/home/saesha/Documents/tango/F5_2/20170131125250.png'
    # '/home/saesha/Documents/tango/F5_3/20170205114543.png'
    # '/home/saesha/Documents/tango/F5_4/20170205115252.png'
    # '/home/saesha/Documents/tango/F5_5/20170205115820.png'
    # '/home/saesha/Documents/tango/F5_6/20170205114156.png'
    # '/home/saesha/Documents/tango/F5_7/20170409113201.png'
    # '/home/saesha/Documents/tango/F5_8/20170409113636.png'
    # '/home/saesha/Documents/tango/F5_9/20170409114748.png'
    # '/home/saesha/Documents/tango/F5_10/20170409115054.png'
    # '/home/saesha/Documents/tango/F5_11/20170409115625.png'
    # '/home/saesha/Documents/tango/F5_12/20170409120348.png'
    # '/home/saesha/Documents/tango/F5_13/20170409120957.png'
    # '/home/saesha/Documents/tango/F5_14/20170409121712.png'

    # # layouts
    # '/home/saesha/Dropbox/myGits/sample_data/HH/HIH/HIH_04.png'
    # '/home/saesha/Dropbox/myGits/sample_data/HH/E5/E5_06.png'
    # '/home/saesha/Dropbox/myGits/sample_data/HH/F5/F5_04.png'
    # '/home/saesha/Dropbox/myGits/sample_data/sweet_home/kpt4a.png'
)

for image_1 in "${file_list[@]}"
do
    for image_2 in "${file_list[@]}"
    do
	echo $image_1 and $image_2
	python align_images.py --img_src $image_1 --img_dst $image_2 --method ECC -save_to_file
	# python align_images.py --img_src $image_1 --img_dst $image_2 --method LK -save_to_file

    done
done
