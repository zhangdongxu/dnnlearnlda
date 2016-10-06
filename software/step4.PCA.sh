cd program/pca
echo 'Start PCA on ' $2 ', dimension is '$1
python pca.py $1 $2
echo 'PCA result is saved in data/'$2'/' 
cd ../../
