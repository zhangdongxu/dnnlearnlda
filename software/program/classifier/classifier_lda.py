from sklearn import svm
import sys
#import sklearn

print 'loading data...'

train_input = open(sys.argv[1])
train_ans = open('data/train.ans')
test_input = open(sys.argv[2])
test_ans = open('data/test.ans')

train = [[],[]]
test = [[],[]]

ans_id = {}
count_ans = 0
for line in train_input:
    data = [float(v) for v in line.split()[1:]]
    s = sum(data)
    for i in range(len(data)):
        data[i]/=s
    train[0].append(data)
for line in train_ans:
    ans = line.split()[1]
    if ans not in ans_id:
        ans_id[ans]=count_ans
        count_ans += 1
    train[1].append(ans_id[ans])
for line in test_input:
    data = [float(v) for v in line.split()[1:]]
    s = sum(data)
    for i in range(len(data)):
        data[i]/=s
    test[0].append(data)
for line in test_ans:
    ans = line.split()[1]
    test[1].append(ans_id[ans])

print 'training...'
lin_clf = svm.LinearSVC()
#lin_clf = svm.SVC(kernel = 'linear')
lin_clf.fit(train[0],train[1])

dec = lin_clf.predict(train[0])
count = 0
for i in range(len(train[0])):
    if train[1][i]== dec[i]:
        count+=1
precision = float(count)/len(train[0])
print 'precision = '+ str(precision)


print 'predicting...'
dec = lin_clf.predict(test[0])

count = 0
for i in range(len(test[0])):
    if test[1][i]== dec[i]:
        count+=1
precision = float(count)/len(test[0])

print 'precision = '+ str(precision)

