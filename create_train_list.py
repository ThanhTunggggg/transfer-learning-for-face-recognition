#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:38:06 2018

@author: thanhtung
"""
f = open("list.txt", "w")
for i in range(54):
    f.write("hd5/train_"+str(i)+".h5\n")
f.close()
