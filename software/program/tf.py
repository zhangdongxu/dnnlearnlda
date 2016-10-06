import math,sys
input = '../data/' + sys.argv[1] + '/train.data'
input2 = '../data/'+ sys.argv[1] + '/test.data'
fout = open('../data/'+ sys.argv[1] + '/train.tf','w')
fout2 = open('../data/'+ sys.argv[1] + '/test.tf','w')
idf={}
num_doc = 0

print 'load data...'

for line in open(input):
    num_doc += 1
    l = line.split()
    docid = l[0]
    word=l[2:]
    for w in word:
        if int(w.split(':')[0]) not in idf:
            idf[int(w.split(':')[0])] = 1

print 'count tf value for training data'
for line in open(input):
    l = line.split()
    docid = l[0]
    word = l[2:]
    totalnum=0
    tfidf=['0.0' for i in range(len(idf))]
    
    for w in word:
        wnum =int(w.split(':')[1])
        totalnum += wnum
    for w in word:
        wid = w.split(':')[0]
        wnum =float(w.split(':')[1])
        tfidf[int(wid)] = str((wnum/totalnum) * idf[int(wid)])
    fout.write(docid + ' ' + ' '.join(tfidf) + '\n')
fout.close()
        
print 'count tf value for test data'
for line in open(input2):
    l = line.split()
    docid = l[0]
    word = l[2:]
    totalnum=0
    tfidf=['0.0' for i in range(len(idf))]
    
    for w in word:
        wnum =int(w.split(':')[1])
        totalnum += wnum
    for w in word:
        wid = w.split(':')[0]
        wnum =float(w.split(':')[1])
        tfidf[int(wid)] = str((wnum/totalnum) * idf[int(wid)])
    fout2.write(docid + ' ' + ' '.join(tfidf) + '\n')
fout2.close()

print 'training and test data are saved into ./data/' + sys.argv[1] + '/'
