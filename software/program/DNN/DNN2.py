'''
build a tweet sentiment analyzer
'''
from collections import OrderedDict
import cPickle as pkl
import random
import sys,os
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_idx(n, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int64")

    if shuffle:
        random.shuffle(idx_list)

    return idx_list


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = {}

    params['U1'] = (1/numpy.sqrt(options['dim_input']) * ( 2 * numpy.random.rand(options['dim_input'],options['dim1']) - 1)).astype(config.floatX)
    params['b1'] = numpy.zeros((options['dim1'],)).astype(config.floatX)

    params['U2'] = (1/numpy.sqrt(options['dim1']) * ( 2 * numpy.random.rand(options['dim1'],options['dim_output']) - 1)).astype(config.floatX)
    params['b2'] = numpy.zeros((options['dim_output'],)).astype(config.floatX)
    
    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)# .get_value()
    return tparams



def sgd(lr, tparams, grads, x, y, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]


    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    f_update = theano.function([lr],[], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(seed=1234)

    x = tensor.dvector('x')
    y = tensor.dvector('y')

    proj1 = tensor.tanh(tensor.dot(x, tparams['U1']) + tparams['b1'])
    proj2 = tensor.nnet.softmax(tensor.dot(proj1, tparams['U2']) + tparams['b2'])
    cost =  tensor.sum(y * tensor.log(y / proj2))
    return  x, y, cost, proj2


def train(
    dim1= 50, 
    dim_output=50, 
    max_epochs=150,  # The maximum number of epoch to run
    dispFreq=100,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.2,  # Learning rate for sgd (not used for adadelta and rmsprop)
    saveto='model.npz',  # The best model will be saved there
    saveFreq=-1,  # Save the parameters after every saveFreq updates
    dataset='imdb',
    data_dir='data',
    output_dir = 'experiment_50_0.2',
    train_ans = 'final.gamma.label_50',
    test_ans = 'infer-gamma.dat_50'
):
    if os.path.exists(output_dir)==False:
        os.mkdir(output_dir)
    # Model options
    model_options = locals().copy()
    print "model options", model_options

    load_data = imdb.load_data

    print 'Loading data'
    train, test = load_data(data_dir=data_dir, 
                             train_input = 'train.tf', 
                             test_input = 'test.tf',
                             train_ans = train_ans,
                             test_ans = test_ans)
   
    id2word = {}
    for line in open(data_dir + '/word2id'):
        id2word[int(line.split()[1])] = line.split()[0]

    dim_input = len(train[0][0])
    model_options['dim_input'] = dim_input
    dim_output = len(train[1][0])
    model_options['dim_output'] = dim_output

    print 'Building model'
    
    params = init_params(model_options)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    (x, y, cost, output) = build_model(tparams, model_options)
    
    #if we set a gaussian priori of softmax weight U, 
    #decay_c is the hyper-parameter to control the penalty strength. 
    #Notice we should not include bias weight.

    grads = tensor.grad(cost, wrt=tparams.values())
    
    f_cost = theano.function([x,y], cost, name='f_cost')# define cost function
    f_output = theano.function([x], output, name= 'f_output')
    
    '''
    id = 0
    for k, p in tparams.iteritems():
        if k  in ['U2','b2']: 
            grads[id]=grads[id]/numpy.sqrt(model_options['dim1'])
        id += 1
    '''
    #f_grad = theano.function([x,x_rev, y], grads, name='f_grad') # for debug

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = sgd(lr, tparams, grads, x, y, cost)# create gradient count function and parameter update function
    
    print 'Optimization'

    kf_test = get_idx(len(test[0]))

    print "%d train examples" % len(train[0])
    print "%d test examples" % len(test[0])
    history_errs = []
    best_p = None
    bad_count = 0


    if saveFreq == -1:
        saveFreq = len(train[0]) 

    uidx = 0  # the number of update done
    estop = False  # early stop
    

    try:
        for eidx in xrange(max_epochs):
            
            
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_idx(len(train[0]), shuffle=True)
            cost_display = 0
            for train_index in kf:
                uidx += 1
                n_samples += 1
                y = train[1][train_index]
                x = train[0][train_index]

                cost = f_grad_shared(x, y)
                #grads = f_grad(x, y)
                f_update(lrate)
                cost_display += cost

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost_display / dispFreq
                    cost_display = 0
                    sys.stdout.flush()


            
            print "Testing, Epoch", eidx,
            start_time = time.clock()
            test_cost = 0
            for test_index in kf_test:
                y = test[1][test_index]
                x = test[0][test_index]
                test_cost += f_cost(x, y)
            end_time = time.clock()
            print '    Cost %f, spend %f sec/epochs' % ( test_cost / len(kf_test)  ,end_time - start_time)

            #fout = open(output_dir+'/'+ str(eidx)+'_result','w')
            #for test_index in kf_test:
            #    x = test[0][test_index]
            #    y = test[1][test_index]
            #    output = f_output(x)
            #    fout.write(' '.join([str(yy) for yy in y])+'\n' + ' '.join([str(oo) for oo in output])+'\n'+'\n')
            #fout.close()
            ########output the predict after each iter#########
            #lrate=lrate/2
        ### saving model ###############
        print 'Saving...',
        params = unzip(tparams)
        numpy.savez(output_dir + '/' + saveto, **params)
        ### saving analysis file #######
        fout_a = open(output_dir+'/analysis.txt','w')
        layer1 = numpy.abs(numpy.tanh(params['U1']+ params['b1']))
        layer2 = numpy.abs(numpy.tanh(numpy.dot(  numpy.tanh(params['U1'] + params['b1'])  ,params['U2']  ) + params['b2'] ))
        
        for id in range(dim1):
            top20_id = numpy.argsort(layer1[:,id])[::-1][:20]
            fout_a.write('1 '+str(id)+' '+' '.join([id2word[w] for w in top20_id]) + '\n')
        for id in range(dim_output):
            top20_id = numpy.argsort(layer2[:,id])[::-1][:20]
            fout_a.write('2 '+str(id)+' '+' '.join([id2word[w] for w in top20_id]) + '\n')
        fout_a.close()
        ### saving features for classification ######
        fout_train = open(output_dir+'/dnn_train_feature','w')
        fout_train2 = open(output_dir+'/lda_train_feature','w')
        for i in range(len(train[0])):
            x = train[0][i]
            output = f_output(x)
            y = train[1][i]
            
            fout_train.write(str(i)+' '+ ' '.join([str(v) for v in output[0]]) + '\n')
            fout_train2.write(str(i)+' '+ ' '.join([str(v) for v in y]) + '\n')
        fout_train.close()
        fout_train2.close()

        fout_test = open(output_dir+'/dnn_test_feature','w')
        fout_test2 = open(output_dir+'/lda_test_feature','w')
        for i in range(len(test[0])):
            x = test[0][i]
            output = f_output(x)
            y = test[1][i]
            
            fout_test.write(str(i+len(train[0])) + ' '+' '.join([str(v) for v in output[0]]) + '\n')
            fout_test2.write(str(i+len(train[0])) + ' ' + ' '.join([str(v) for v in y]) + '\n')
        fout_test.close()
        fout_test2.close()
    except KeyboardInterrupt:
        print "Training interupted"



if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train(
    dim1= 2*int(sys.argv[1]), 
    dim_output=int(sys.argv[1]), 
    max_epochs=300,  # The maximum number of epoch to run
    dispFreq=100,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=float(sys.argv[2]),  # Learning rate for sgd (not used for adadelta and rmsprop)
    saveto='model.npz',  # The best model will be saved there
    saveFreq=-1,  # Save the parameters after every saveFreq updates
    dataset='imdb',
    data_dir=sys.argv[3],
    output_dir = sys.argv[4],
    train_ans = 'train.lda_' + sys.argv[1],
    test_ans = 'test.lda_' + sys.argv[1])
