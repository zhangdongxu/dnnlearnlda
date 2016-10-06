import os,numpy

def load_data(data_dir="data", train_input = 'train.tfidf', test_input = 'test.tfidf', train_ans = 'final.gamma.label', test_ans = 'infer-gamma.dat'):

    #############
    # LOAD DATA #
    #############
    if os.path.exists(data_dir)==False:
        os.mkdir(data_dir)
      
    train_set = ([],[])# train_set[0][i] is a matrix, each row is an article, each column is a tfidf 
                       # train_set[1][i] is a matrix, each row is an article with distribution on topics.
    test_set = ([],[]) # test_set[0][i] is a matrix, each row is an article, each column is a tfidf 
                       # test_set[1][i] is a matrix, each row is an article with distribution on topics.

    for line in open(data_dir+ '/'+ train_input):
            l = line.split()
            docid = l[0]
            tfidf = [float(v) for v in l[1:]]
            train_set[0].append(tfidf)
    for line in open(data_dir+ '/'+ train_ans):
            l = line.split()
            docid = l[0]
            distribution = [float(v) for v in l[1:]]
            sum_ = sum(distribution)
            distribution_ = [v/sum_ for v in distribution]
            train_set[1].append(distribution_)

    for line in open(data_dir+ '/'+ test_input):
            l = line.split()
            docid = l[0]
            tfidf = [float(v) for v in l[1:]]
            test_set[0].append(tfidf)
    for line in open(data_dir+ '/'+ test_ans):
            l = line.split()
            docid = l[0]
            distribution = [float(v) for v in l[1:]]
            sum_ = sum(distribution)
            distribution_ = [v/sum_ for v in distribution]
            test_set[1].append(distribution_)


    return train_set, test_set

def load_data_finetune(data_dir="data", train_input = 'train.tfidf', test_input = 'test.tfidf', train_ans = 'final.gamma.label', test_ans = 'infer-gamma.dat'):

    #############
    # LOAD DATA #
    #############
    if os.path.exists(data_dir)==False:
        os.mkdir(data_dir)
      
    train_set = ([],[])# train_set[0][i] is a matrix, each row is an article, each column is a tfidf 
                       # train_set[1][i] is a matrix, each row is an article with distribution on topics.
    test_set = ([],[]) # test_set[0][i] is a matrix, each row is an article, each column is a tfidf 
                       # test_set[1][i] is a matrix, each row is an article with distribution on topics.
    ans_id = {}
    count_ans = 0
    for line in open(data_dir+ '/'+ train_input):
        l = line.split()
        docid = l[0]
        tfidf = [float(v) for v in l[1:]]
        train_set[0].append(tfidf)
    for line in open(data_dir+ '/'+ train_ans):
        ans = line.split()[1]
        if ans not in ans_id:
            ans_id[ans]=count_ans
            count_ans += 1
        train_set[1].append(ans_id[ans])

    for line in open(data_dir+ '/'+ test_input):
        l = line.split()
        docid = l[0]
        tfidf = [float(v) for v in l[1:]]
        test_set[0].append(tfidf)
    for line in open(data_dir+ '/'+ test_ans):
        ans = line.split()[1]
        test_set[1].append(ans_id[ans])


    return train_set, test_set

def load_data_cls(data_dir="data", train_input = 'train.tfidf', test_input = 'test.tfidf', train_ans = 'final.gamma.label', test_ans = 'infer-gamma.dat'):

    #############
    # LOAD DATA #
    #############
    if os.path.exists(data_dir)==False:
        os.mkdir(data_dir)
      
    train_set = ([],[])# train_set[0][i] is a matrix, each row is an article, each column is a tfidf 
                       # train_set[1][i] is a matrix, each row is an article with distribution on topics.
    test_set = ([],[]) # test_set[0][i] is a matrix, each row is an article, each column is a tfidf 
                       # test_set[1][i] is a matrix, each row is an article with distribution on topics.
    ans_id = {}
    count_ans = 0
    for line in open(data_dir+ '/'+ train_input):
        train_set[0].append([float(v) for v in line.split()[1:]])
    for line in open(data_dir+ '/'+ train_ans):
        ans = line.split()[1]
        if ans not in ans_id:
            ans_id[ans]=count_ans
            count_ans += 1
        train_set[1].append(ans_id[ans])
    for line in open(data_dir+ '/'+ test_input):
        test_set[0].append([float(v) for v in line.split()[1:]])
    for line in open(data_dir+ '/'+ test_ans):
        ans = line.split()[1]
        if ans not in ans_id:
            ans_id[ans]=count_ans
            count_ans += 1
        test_set[1].append(ans_id[ans])


    return train_set, test_set


