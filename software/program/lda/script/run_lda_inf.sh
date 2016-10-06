#!/bin/sh
num=$1
settings=inf-settings.txt
data=../../../data/$2/test.data
model=../../../data/$2/lda_result_${num}/final

directory=../../../data/$2/lda_infer_${num}
mkdir $directory

name=../../../data/$2/lda_infer_${num}/infer

echo 'Start inferring...'

a=$(date "+%Y-%m-%d %k:%M:%S")
./lda "inf" $settings $model $data $name
b=$(date "+%Y-%m-%d %k:%M:%S")


echo 'Start time: '$a', End time:' $b
