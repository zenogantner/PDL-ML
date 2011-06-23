#!/bin/sh

# Get example datasets
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/eunite2001
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/eunite2001.t

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.bz2
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.t.bz2
bunzip2 leu.bz2
bunzip2 leu.t.bz2

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1.t

http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide3
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide3.t