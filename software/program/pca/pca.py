from sklearn.decomposition import PCA
import sys
#X = [[1,2,3],[2,3,4],[2,1,1]]
#X_t = [[2,1,1]]

dim=int(sys.argv[1])

print 'load data...'
X = []
for line in open('../../data/' + sys.argv[2] + '/train.tf'):
    X.append([float(v) for v in line.split()[1:]])

X_t = []
for line in open('../../data/'+ sys.argv[2]+'/test.tf'):
    X_t.append([float(v) for v in line.split()[1:]])

print 'pca...'
pca = PCA(n_components=dim)
trans = pca.fit(X)

print 'transform...'
X_train = trans.transform(X)
X_test  = trans.transform(X_t)

print 'output...'
fout1 = open('../../data/' + sys.argv[2]+'/train.pca_'+str(dim),'w')
for i in range(len(X_train)):
    fout1.write(str(i)+' '+ ' '.join( [str(v) for v in X_train[i]] ) + '\n') 
fout1.close()

fout2 = open('../../data/' + sys.argv[2]+'/test.pca_'+str(dim),'w')
for i in range(len(X_test)):
    fout2.write(str(i + len(X_train))+' '+ ' '.join( [str(v) for v in X_test[i]] ) + '\n')
fout2.close()

