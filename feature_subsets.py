#! /usr/bin/python2

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:45:41 2014

@author: cheesinglee
"""

import csv
import argparse
from pprint import pprint
from copy import copy
from os.path import join,split,splitext

from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from bigml.api import BigML

# default values

# number of cross-validation folds for evaluation
N_FOLDS = 5

# search staleness
K = 5

# difference needed to become new best node
EPSILON = 0.001

# per feature score penalty
PENALTY = 0.001

def generate_cross_validation(filename,n_folds):
    """
    use sklearn kfolds class to create kfold cross validation data sources
    """

    # read the sourcefile
    with open(filename) as source_file:
        reader = csv.reader(source_file)
        fieldnames = reader.next()
        data = [row for row in reader]

    # extract target labels and transform to sklearn format
    labels = [row[-1] for row in data]
    lenc = LabelEncoder()
    Y = lenc.fit_transform(labels)

    # create iterable to generate folds
    kfolds = StratifiedKFold(Y,n_folds)

    # iterate over folds and write CSV files
    n = 1
    (head,tail) = split(filename)
    template = join(head,splitext(tail)[0])+'_{t}_{num}.csv'
    fold_filenames = []
    for idx_train,idx_test in kfolds:
        data_train = [data[i] for i in idx_train]
        data_test = [data[i] for i in idx_test]

        filename_train = template.format(t='train',num=str(n))
        filename_test = template.format(t='test',num=str(n))
        fold_filenames.append((filename_train,filename_test))
        n += 1

        with open(filename_train,'w') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(fieldnames)
            writer.writerows(data_train)

        with open(filename_test,'w') as test_file:
            writer = csv.writer(test_file)
            writer.writerow(fieldnames)
            writer.writerows(data_test)

    return fold_filenames

def expand_state(parent):
    """ get all connected nodes """
    children = []
    for i in range(len(parent)):
        child = copy(parent)
        child[i] = not child[i]
        children.append(child)
    return children

def evaluate(input_fields,cv_datasets,api,penalty):
    """ do cross-validation using the given feature subset """
    args = {'input_fields':input_fields}

    models = []
    for (train,test) in cv_datasets:
        m = api.create_model(train,args)
        models.append(m)

    accuracy_scores = []
    for (i,(train,test)) in enumerate(cv_datasets):
        e = api.create_evaluation(models[i],test)
        e = api.check_resource(e,api.get_evaluation)
        accuracy_scores.append(e['object']['result']['model']['accuracy'])

    return (sum(accuracy_scores)/len(accuracy_scores) - penalty*len(input_fields))

def find_max_state(states):
    maxval = -1
    maxstate = ''
    for (v,f) in states:
        if f > maxval:
            maxstate = v
            maxval = f
    return maxstate,maxval

def main(args):
    print('initialize BigML API')
    if args.username and args.apikey:
        api = BigML(args.username,args.apikey)
    else:
        api = BigML()

    print('generate cross validation splits')
    cv_files = generate_cross_validation(args.filename,args.nfolds)

    cv_datasets = []
    for (train_file,test_file) in cv_files:
        train_source = api.create_source(train_file)
        test_source = api.create_source(test_file)

        train_dataset = api.create_dataset(train_source)
        test_dataset = api.create_dataset(test_source)
        cv_datasets.append((train_dataset,test_dataset))


    # wait for dataset creation to finish so we can find out the number of features
    dataset_res = api.check_resource(cv_datasets[0][0],api.get_dataset)
    dataset_obj = dataset_res['object']

    # initial feature set
    field_ids = dataset_obj['fields'].keys()
    field_ids.remove(dataset_obj['objective_field']['id'])
    initial_state = [False for id in field_ids]

    # do best-first search
    done = False
    open_list = [(initial_state,0)]
    closed_list = []
    best_accuracy = -1
    best_unchanged_count = 0
    while not done:
        (v,fv) = find_max_state(open_list)
        v_ids = [field_ids[i] for (i,val) in enumerate(v) if val]
        print('Max state is: %s\n Accuracy = %f' % (v_ids,fv))
        closed_list.append((v,fv))
        open_list.remove((v,fv))
        if (fv - EPSILON) > best_accuracy:
            best_state = v
            best_accuracy = fv
            best_unchanged_count = 0
            print('new best state')
        else:
            best_unchanged_count += 1

        children = expand_state(v)
        for c in children:
            if (c not in [pair[0] for pair in open_list]
            and c not in [pair[0] for pair in closed_list]):
                input_fields = [id for (i,id) in enumerate(field_ids) if c[i]]
                print('Evaluating %s' % input_fields)
                val = evaluate(input_fields,cv_datasets,api,args.penalty)
                open_list.append((c,val))

        if best_unchanged_count >= args.staleness:
            done = True

    best_features = [field_ids[i] for (i,val) in enumerate(best_state) if val]
    print('The best feature subset is: %s \n Accuracy = %0.2f%%' % (best_features,best_accuracy*100))
    print('Evaluated %d/%d feature subsets' % ((len(open_list) + len(closed_list)),2**len(field_ids)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',help='path to CSV file')
    parser.add_argument('-u','--username',type=str,help='BigML username')
    parser.add_argument('-a','--apikey',type=str,help='BigML API key')
    parser.add_argument('-n','--nfolds',type=int,help='Number of cross-validation folds [default=%d]' % N_FOLDS,default=N_FOLDS)
    parser.add_argument('-k','--staleness',type=int,default=K,help='Staleness parameter for best-first search [default=%d]' % K)
    parser.add_argument('-p','--penalty',type=float,default=PENALTY,help='Per-feature penalty factor [default=%0.3f]' % PENALTY)
    args = parser.parse_args()
    main(args)
