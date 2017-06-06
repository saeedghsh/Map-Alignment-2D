import re
import numpy as np
import matplotlib.pyplot as plt


########################################
##################### match score matrix
########################################
if 1:
    scores = np.ones((36,36))
    adr = '/home/saesha/Dropbox/myGits/orebro_visit/map_alignment_paper/figures/sensor_sensor_result/'
    fle = 'sensos_sensor_matchscore'

    f = open(adr + fle + '.txt', 'r')
    for line in f:
        spl = re.findall(r"[\w']+", line)
        if len(spl) > 0:
            idx1, idx2, score = int(spl[0]), int(spl[1]), float(spl[-1])/10000.
            scores[idx1, idx2] = score
            scores[idx2, idx1] = score
            # print (idx1, idx2, score)


    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(scores, cmap=plt.cm.gray, interpolation=None)
    ax.matshow(scores, cmap=plt.cm.gray, vmin=0, vmax=1)
    for n in [3.5, 7.5, 21.5]:
        ax.plot([n, n], [-.5, 35.5], 'r')
        ax.plot([-.5, 35.5], [n, n], 'r')
        
    # ax.margins(0.2)
    # ax.set_axis_off()

    plt.xticks( [1.5, 5.5, 14.5, 28.5], ['HIH','KPT4A', 'E5','F5'], horizontalalignment='center')
    plt.yticks( [1.5, 5.5, 14.5, 28.5], ['HIH','KPT4A', 'E5','F5'], rotation='vertical', verticalalignment='center')


    for i in range(36):
        n = i-.5
        ax.plot([n, n], [n, n+1], 'b')
        ax.plot([n, n+1], [n, n], 'b')
        ax.plot([n+1, n+1], [n, n+1], 'b')
        ax.plot([n, n+1], [n+1, n+1], 'b')

    
    # ax.text(.02, 1.05, 'HIH', transform=ax.transAxes)
    # ax.text(.11, 1.05, 'KPT4A', transform=ax.transAxes)
    # ax.text(.4, 1.05, 'E5', transform=ax.transAxes)
    # ax.text(.8, 1.05, 'F5', transform=ax.transAxes)
    
    # ax.text(-.05, .98, 'HIH', rotation='vertical', transform=ax.transAxes)
    # ax.text(-.05, .89, 'KPT4A', rotation='vertical', transform=ax.transAxes)
    # ax.text(-.05, .6, 'E5', rotation='vertical', transform=ax.transAxes)
    # ax.text(-.05, .2, 'F5', rotation='vertical', transform=ax.transAxes)


    if 1: # marking succes of alignment
        adr = '/home/saesha/Dropbox/myGits/orebro_visit/map_alignment_paper/figures/sensor_sensor_result/'
        idx_bias = 0

        for fle,dim in [('success_HIH',4), ('success_KPT4A',4), ('success_E5',14), ('success_F5',14)]:
            f = open(adr + fle + '.txt', 'r')
            for line in f:
                spl = re.findall(r"[\w']+", line)
                if len(spl) > 0:
                    idx1, idx2 = int(spl[0])-1, int(spl[1])-1
                    success = float(spl[2]) if float(spl[2])<=1 else float(spl[2])/10.
                    if success > 0:
                        ax.plot(idx_bias+idx1, idx_bias+idx2, 'g.')
                        ax.plot(idx_bias+idx2, idx_bias+idx1, 'g.')
                    else:
                        ax.plot(idx_bias+idx1, idx_bias+idx2, 'r.')
                        ax.plot(idx_bias+idx2, idx_bias+idx1, 'r.')

            idx_bias += dim            
    
    if 1:
        plt.savefig(adr + fle + '.png', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


########################################
######################### success matrix
########################################
if 0:
    adr = '/home/saesha/Dropbox/myGits/orebro_visit/map_alignment_paper/figures/sensor_sensor_result/'
    fle,dim = [('success_E5',14),
               ('success_F5',14),
               ('success_HIH',4),
               ('success_KPT4A',4)][0]

    successes = np.ones((dim,dim))
    f = open(adr + fle + '.txt', 'r')
    for line in f:
        spl = re.findall(r"[\w']+", line)
        if len(spl) > 0:
            idx1, idx2 = int(spl[0])-1, int(spl[1])-1
            success = float(spl[2]) if float(spl[2])<=1 else float(spl[2])/10.
            successes[idx1, idx2] = success
            successes[idx2, idx1] = success

    
    fig, ax = plt.subplots(figsize=(dim/2., dim/2.))
    # ax.imshow(successes, cmap=plt.cm.gray, interpolation=None)
    ax.matshow(successes, cmap=plt.cm.gray, vmin=0, vmax=1)

    plt.xticks( range(dim), range(1,dim+1) )
    plt.yticks( range(dim), range(1,dim+1) )

    for i in range(dim):
        n = i-.5
        ax.plot([n, n], [n, n+1], 'b')
        ax.plot([n, n+1], [n, n], 'b')
        ax.plot([n+1, n+1], [n, n+1], 'b')
        ax.plot([n, n+1], [n+1, n+1], 'b')


    # for n in [3.5, 7.5, 21.5]:
    #     ax.plot([n, n], [-.5, 35.5], 'r')
    #     ax.plot([-.5, 35.5], [n, n], 'r')
        
    # # ax.margins(0.2)
    # ax.set_axis_off()
    
    if 1:
        plt.savefig(adr + fle + '.png', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

