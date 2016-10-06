import numpy

id2word = {}
for line in open('word2id'):
    id2word[int(line.split()[1])]=line.split()[0]

i = 0
for line in open('final.beta'):
    i+=1
    a = numpy.argsort([float(v) for v in line.split()])[::-1][:20]
    print str(i)+' ' +' '.join([id2word[w] for w in a])
