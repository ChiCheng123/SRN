#!/bin/bash
VERSION=`nvcc --version | tail -1 | grep -oE "release [0-9\.]{1,10}"`
echo "using cuda version:$VERSION"
MAJOR=`nvcc --version | tail -1 | grep -oE "release [0-9]{1,10}" | grep -oE "[0-9]{1,10}"`
arch="-gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61"
arch7=" -gencode arch=compute_70,code=sm_70"
if [ $MAJOR -gt 8 ]; then
    arch="$arch $arch7"
fi
echo "compiling for architecture: $arch"


for file in ./*
do
    if test -d $file && test -f $file/build.sh
    then
        cd $file
        echo building $file
        bash build.sh "$arch"
        if [ $? != 0 ]; then
            exit
        fi
        cd ..
    fi
done

