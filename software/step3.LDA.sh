cd program/lda/script
sh run_lda.sh $1 $2 
sh run_lda_inf.sh $1 $2
cd ../../../
cp data/$2/lda_result_$1/final.gamma.labeld data/$2/train.lda_$1
cp data/$2/lda_infer_$1/infer-gamma.dat data/$2/test.lda_$1
