from itertools import combinations
import numpy as np
import random

DATASET_DIR = '/Users/bjuncklaus/Dropbox/Evolutionary/Data/Four_ants_3C286/'
# DATASET_DIR = '/users/bmartins/datasets/'
MS = 'Four_ants_3C286.ms'
# MS = '/users/rurvashi/SummerProjects/2016/Data/Only_spw11_From_3C286mos.ms/'
# MS = '/home/vega/rurvashi/RSRO/SNR/CBand_G29.7-0.3/G29.7-0.3_cal.ms'
DDID_TABLE = MS + '/DATA_DESCRIPTION'
SPEC = MS+'/SPECTRAL_WINDOW'
POL = MS+'/POLARIZATION'

file_training = open(DATASET_DIR + 'original_data.csv', 'w')
# file_training.close()

# file_training = open('original_data.csv', 'a')
#file_training = open(DATASET_DIR + 'testing.csv', 'w')

antennas = []
ignored_antennas = []
# ignored_antennas = [16,19, 25]
# for i in range(26):
#     if (i not in ignored_antennas):
#         antennas.append(i)
antennas = [0,1,2,3]
antenna_combinations = list(combinations(antennas, 2))

print("Writing to", file_training.name)
DDID = 0
#scans = [37,55,56,57,79,80,81,82,100,126,128,145,147,148,149,169,170,172,173,190,191,192,194,214,215,216,217]

scans = [30, 31]
for scan in scans:
    for data_description_id in range(16):
        if data_description_id != 8:
            for antenna_pair in antenna_combinations:
                tb.open(MS)

                query = 'DATA_DESC_ID=={0} && SCAN_NUMBER=={1} && ANTENNA1=={2} && ANTENNA2=={3}'.format(data_description_id, scan, antenna_pair[0], antenna_pair[1])

                tb1 = tb.query(query)
                data = tb1.getcol('DATA')
                times = tb1.getcol('TIME')  # in seconds, in MJD.  Can convert to actual time.
                flags = tb1.getcol('FLAG')

                tb1.close()
                tb.close()


                shp = data.shape
                print("scan: ", scan, "data shape:", data.shape, "antenna pair:", antenna_pair, "SPW:", data_description_id)

                ### Gather metadata for spw 9***
                # DDID = 4

                freqlabels = np.zeros( shp[1] )

                ## To get Frequencies

                tb.open(DDID_TABLE)
                spec_ddids = tb.getcol('SPECTRAL_WINDOW_ID')
                tb.close()

                tb.open(SPEC)
                chanfreqs = tb.getcell( 'CHAN_FREQ', spec_ddids[data_description_id])
                tb.close()


                ## To get polarization labels

                tb.open(DDID_TABLE)
                pol_ddids = tb.getcol('POLARIZATION_ID')
                tb.close()

                tb.open(POL)
                pols = tb.getcell( 'CORR_TYPE', pol_ddids[data_description_id])
                tb.close()

                for polarization in range(len(pols)):
                    j = 0
                    for time in range(len(times)):
                        #file_training.write(str(antenna_pair[0]))
                        #file_training.write(',')
                        #file_training.write(str(antenna_pair[1]))
                        #file_training.write(',')
                        file_training.write(str(times[time]))
                        file_training.write(',')

                        file_training.write(str(pols[polarization]))
                        file_training.write(',')

                        for frequency in chanfreqs:
                            file_training.write(str(frequency))
                            file_training.write(',')

                        for k in range(64):
                            file_training.write(str(np.abs(data[polarization][k][j])))
                            file_training.write(',')

                        j += 1

                        i = 0
                        for flag in flags[polarization]:
                            file_training.write(str(int(flag[time] == True)))

                            if (i < (len(flags[polarization])-1)):
                                file_training.write(',')

                            i += 1

                        file_training.write('\n')

file_training.close()