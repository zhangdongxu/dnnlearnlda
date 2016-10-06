K=$1
corpus=$2

cp data/$2/train.ans program/classifier/data/
cp data/$2/test.ans program/classifier/data/
cp data/$2/train.pca_$1 program/classifier/data/
cp data/$2/test.pca_$1 program/classifier/data/
cp data/$2/train.lda_$1 program/classifier/data/
cp data/$2/test.lda_$1 program/classifier/data/
cp data/$2/experiment_2-layer_DNN_$1/dnn_train_feature program/classifier/data/train.DNN2_$1
cp data/$2/experiment_2-layer_DNN_$1/dnn_test_feature program/classifier/data/test.DNN2_$1
cp data/$2/experiment_3-layer_DNN_$1/dnn_train_feature program/classifier/data/train.DNN3_$1
cp data/$2/experiment_3-layer_DNN_$1/dnn_test_feature program/classifier/data/test.DNN3_$1

cd program/classifier/

echo 'Accuracy of PCA'
python classifier.py data/train.pca_$1 data/test.pca_$1
echo 'Accuracy of LDA'
python classifier.py data/train.lda_$1 data/test.lda_$1
echo 'Accuracy of two layer DNN'
python classifier.py data/train.DNN2_$1 data/test.DNN2_$1
echo 'Accuracy of three layer DNN'
python classifier.py data/train.DNN3_$1 data/test.DNN3_$1



