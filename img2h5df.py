import cv2 as cv
import numpy as np
import h5py
import os
import random

f = open('data_clean.txt')
A = []
for line in f:
    line = line.strip()
    A.append(line)
f.close()

random.shuffle(A)
IMG = []
LAB = []
num = 0
cnt = 0
for i in range(len(A)):
    line = A[i]
    # print (line)
    # input()
    path = "dataset/" + line.split(' ')[0]
    label = int(line.split(' ')[-1])
    try:
        img = cv.imread(path)
        img = cv.resize(img,(122,144))
        M2 = np.float32([[1,0,11],[0,1,0]])
        img = cv.warpAffine(img,M2,(144,144))
        # cv.imshow('img', img)
        # cv.waitKey(0)
        IMG.append(img)
        LAB.append(label)
        j = cnt+1
        cnt += 1 
        if (j%5000 == 0):
            print ('convet image to hd5 file...')
            IMG = np.array(IMG)
            LAB = np.array(LAB)
            print (IMG.shape)
            print (IMG.shape)
    
            h5_path = os.path.join('hd5/', 'train_'+str(num)+'.h5')
            # with h5py.File(h5_path, 'w') as f:
            # 	f['data'] = IMG
            # 	f['label'] = LAB
            f = h5py.File(h5_path, 'w')
            f.create_dataset('data', data=IMG, compression='gzip', compression_opts=4)
            f.create_dataset('label', data=LAB, compression='gzip', compression_opts=4)
            f.close()
            num += 1
            IMG = []
            LAB = []
            print("Created ./hd5/train_" +str(num)+ ".h5")

        if (cnt%100 == 0):
            print (str(cnt) + '\t' + str(num))
    except:
        print("Cant load image in {}".format(path))


if(len(IMG) != 0):
    IMG = np.array(IMG)
    LAB =np.array(LAB)
  
    h5_path = os.path.join('./hd5/', 'train_'+str(num)+'.h5')
    f = h5py.File(h5_path, 'w')
    f.create_dataset('data', data=IMG, compression='gzip', compression_opts=4)
    f.create_dataset('label', data=LAB, compression='gzip', compression_opts=4)
    f.close()
print("Done.")
