#!/bin/sh
alpha=1
k=$1
settings=settings.txt
data=../../../data/$2/train.data
directory=../../../data/$2/lda_result_${k}
mkdir $directory

echo 'Start LDA on ' $2  ', topic num:' $k 

./lda "est" $alpha $k $settings $data "random" $directory

echo 'LDA model saved in ./data/' $2 '/lda_result_' $k '/'
