#!/bin/bash -ue

file_list=(
    '/home/saesha/Documents/tango/kpt4a_f/20170131163311.png'
    '/home/saesha/Documents/tango/kpt4a_kb/20170131163634.png'
    '/home/saesha/Documents/tango/kpt4a_kl/20170131162628.png'
    '/home/saesha/Documents/tango/kpt4a_lb/20170131164048.png'
    
    '/home/saesha/Documents/tango/HIH_01_full/20170131135829.png'
    
    '/home/saesha/Documents/tango/E5_1/20170131150415.png'
    '/home/saesha/Documents/tango/E5_2/20170131131405.png'
    '/home/saesha/Documents/tango/E5_3/20170131130616.png'
    '/home/saesha/Documents/tango/E5_4/20170131122040.png'
    '/home/saesha/Documents/tango/E5_5/20170205104625.png'
    '/home/saesha/Documents/tango/E5_6/20170205105917.png'
    '/home/saesha/Documents/tango/E5_7/20170205111301.png'
    '/home/saesha/Documents/tango/E5_8/20170205112339.png'
    '/home/saesha/Documents/tango/E5_9/20170205110552.png'
    '/home/saesha/Documents/tango/E5_10/20170205111807.png'

    '/home/saesha/Documents/tango/F5_1/20170131132256.png'
    '/home/saesha/Documents/tango/F5_2/20170131125250.png'
    '/home/saesha/Documents/tango/F5_3/20170205114543.png'
    '/home/saesha/Documents/tango/F5_4/20170205115252.png'
    '/home/saesha/Documents/tango/F5_5/20170205115820.png'
    '/home/saesha/Documents/tango/F5_6/20170205114156.png'

    '/home/saesha/Dropbox/myGits/sample_data/HH/HIH/HIH_04.png'
    '/home/saesha/Dropbox/myGits/sample_data/HH/E5/E5_06.png'
    '/home/saesha/Dropbox/myGits/sample_data/HH/F5/F5_04.png'
    '/home/saesha/Dropbox/myGits/sample_data/sweet_home/kpt4a.png'

)

for file_name in "${file_list[@]}"
do
    echo processing file $file_name ...
    python get_dis_image.py --s $file_name
done
