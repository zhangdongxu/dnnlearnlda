echo 'Learning LDA with '$1 'topics using 2-layer DNN...'
cd program/DNN/
NO_FLAGS="floatX=float64,blas.ldflags=" python DNN2.py $1 $2 ../../data/$3  ../../data/$3/experiment_2-layer_DNN_$1
cd ../../
