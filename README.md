# Description

c++ implementation of [CapsNets White Paper](https://arxiv.org/pdf/1710.09829.pdf)
(for fun, not intending to clean-up)

# Dependencies

openCV 4+

# Build Instructions

modify CMakeLists.txt for OpenCV location
```
mkdir build & cd build
cmake ..
make
```

# Run Instructions

modify main.cpp as desired. see config #define's atop main.cpp

## Example train
```
bash$ ./PANTHRO train
 training network using 500 images, 40 iterations...
 Run 0 MSErrors:  avg= 0.007902  stdev=0.002082
 Run 1 MSErrors:  avg= 0.007686  stdev=0.002037
 Run 2 MSErrors:  avg= 0.007394  stdev=0.001981
 etc...
```

## Example test
```
bash$ ./PANTHRO test
 completed 101 of 601 (numIncorrect: 0)
 completed 201 of 601 (numIncorrect: 0)
 completed 301 of 601 (numIncorrect: 0)
 completed 401 of 601 (numIncorrect: 0)
 completed 501 of 601 (numIncorrect: 0)
 completed 601 of 601 (numIncorrect: 20)
 # Runs: 601
 % Corr: 96%
 mean sq err(avg, stddev): 0.0123916, 0.0127018
 image index start offset: 0
```

## Example: Image Reconstruction
```
bash$ ./PANTHRO image
 enter image index: 22
^C
```
output: ![](/data/for-readme.png "")
 
# Notes

I only trained on images 0..500 of the training set.<br/>
Image & Matrix sizes are hardcoded throughout (as defined in whitepaper)<br/>
Not thread-safe but, if adding, need to update digicaps.cpp + reconstruct.cpp for *weights vars<br/>
<br/>

neat: ![neat](/data/for-readme-chart.png "neat")

