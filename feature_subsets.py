# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:45:41 2014

@author: cheesinglee
"""

import csv
from pprint import pprint
from copy import copy
from os.path import join,split,splitext

from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from bigml.api import BigML

def generate_cross_validation(filename,n_folds):
    """
    use sklearn kfolds class to create kfold cross validation data sources
    """

    # read the sourcefile
    with open(filename) as source_file:
#        reader = DictReader(source_file)
        reader = csv.reader(source_file)
        fieldnames = reader.next()
        data = [row for row in reader]
#        fieldnames = reader.fieldnames

    # extract target labels and transform to sklearn format
#    label_key = fieldnames[-1]
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
#            writer = DictWriter(train_file,fieldnames)
            writer = csv.writer(train_file)
            writer.writerow(fieldnames)
#            writer.writeheader()
            writer.writerows(data_train)

        with open(filename_test,'w') as test_file:
#            writer = DictWriter(test_file,fieldnames)
#            writer.writeheader()
            writer = csv.writer(test_file)
            writer.writerow(fieldnames)
            writer.writerows(data_test)

    return fold_filenames

def expand_state(parent):
    children = []
    for i in range(len(parent)):
        child = copy(parent)
        child[i] = not child[i]
        children.append(child)
    return children

def evaluate(input_fields,cv_datasets):
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

    return (sum(accuracy_scores)/len(accuracy_scores) - PENALTY*len(input_fields))

def find_max_state(states):
    maxval = -1
    maxstate = ''
    for (v,f) in states:
        if f > maxval:
            maxstate = v
            maxval = f
    return maxstate,maxval


USERNAME='cheesinglee'
API_KEY='c057f0dfbc5196f366bc18f9006c5207154e0d91'
api = BigML(USERNAME,API_KEY)

N_FOLDS = 5
K = 5
EPSILON = 0.001
PENALTY = 0.001

#sourcefile = 'data/crx.csv'
#
#print 'generate cross validation splits'
#cv_files = generate_cross_validation(sourcefile,N_FOLDS)
#
#cv_datasets = []
#for (train_file,test_file) in cv_files:
#    train_source = api.create_source(train_file)
#    test_source = api.create_source(test_file)
#
#    train_dataset = api.create_dataset(train_source)
#    test_dataset = api.create_dataset(test_source)
#    cv_datasets.append((train_dataset,test_dataset))
#

cv_datasets = [({'code': 201, 'resource': 'dataset/5306a77e11de500694000000', 'location': 'http://bigml.io/andromeda/dataset/5306a77e11de500694000000', 'object': {'size': 27189, 'code': 201, 'locale': 'en-US', 'updated': '2014-02-21T01:10:22.622994', 'private': True, 'fields_meta': {'count': 0, 'total': 0, 'limit': 1000, 'offset': 0}, 'number_of_predictions': 0, 'category': 0, 'all_fields': True, 'replacements': None, 'source': 'source/5306a77b11de5006ba000000', 'number_of_batchpredictions': 0, 'user_metadata': {}, 'field_types': {'categorical': 0, 'text': 0, 'preferred': 0, 'datetime': 0, 'numeric': 0, 'total': 0}, 'number_of_models': 0, 'number_of_ensembles': 0, 'columns': 0, 'status': {'message': 'The dataset is being processed and will be created soon', 'code': 1}, 'sample_rates': None, 'description': u'', 'tags': [], 'price': 0.0, 'excluded_fields': [], 'ranges': None, 'credits': 0.02592945098876953, 'rows': 0, 'shared': False, 'subscription': True, 'resource': 'dataset/5306a77e11de500694000000', 'name': "crx_train_1's dataset", 'created': '2014-02-21T01:10:22.622948', 'source_status': True, 'number_of_evaluations': 0, 'seeds': None, 'term_limit': 1000}, 'error': None}, {'code': 201, 'resource': 'dataset/5306a78011de5006c1000000', 'location': 'http://bigml.io/andromeda/dataset/5306a78011de5006c1000000', 'object': {'size': 6828, 'code': 201, 'locale': 'en-US', 'updated': '2014-02-21T01:10:24.991141', 'private': True, 'fields_meta': {'count': 0, 'total': 0, 'limit': 1000, 'offset': 0}, 'number_of_predictions': 0, 'category': 0, 'all_fields': True, 'replacements': None, 'source': 'source/5306a77c11de5006b0000000', 'number_of_batchpredictions': 0, 'user_metadata': {}, 'field_types': {'categorical': 0, 'text': 0, 'preferred': 0, 'datetime': 0, 'numeric': 0, 'total': 0}, 'number_of_models': 0, 'number_of_ensembles': 0, 'columns': 0, 'status': {'message': 'The dataset is being processed and will be created soon', 'code': 1}, 'sample_rates': None, 'description': u'', 'tags': [], 'price': 0.0, 'excluded_fields': [], 'ranges': None, 'credits': 0.006511688232421875, 'rows': 0, 'shared': False, 'subscription': True, 'resource': 'dataset/5306a78011de5006c1000000', 'name': "crx_test_1's dataset", 'created': '2014-02-21T01:10:24.991084', 'source_status': True, 'number_of_evaluations': 0, 'seeds': None, 'term_limit': 1000}, 'error': None}), ({'code': 201, 'resource': 'dataset/5306a787714577095a000000', 'location': 'http://bigml.io/andromeda/dataset/5306a787714577095a000000', 'object': {'size': 27205, 'code': 201, 'locale': 'en-US', 'updated': '2014-02-21T01:10:31.367297', 'private': True, 'fields_meta': {'count': 0, 'total': 0, 'limit': 1000, 'offset': 0}, 'number_of_predictions': 0, 'category': 0, 'all_fields': True, 'replacements': None, 'source': 'source/5306a78111de5006aa000000', 'number_of_batchpredictions': 0, 'user_metadata': {}, 'field_types': {'categorical': 0, 'text': 0, 'preferred': 0, 'datetime': 0, 'numeric': 0, 'total': 0}, 'number_of_models': 0, 'number_of_ensembles': 0, 'columns': 0, 'status': {'message': 'The dataset is being processed and will be created soon', 'code': 1}, 'sample_rates': None, 'description': u'', 'tags': [], 'price': 0.0, 'excluded_fields': [], 'ranges': None, 'credits': 0.02594470977783203, 'rows': 0, 'shared': False, 'subscription': True, 'resource': 'dataset/5306a787714577095a000000', 'name': "crx_train_2's dataset", 'created': '2014-02-21T01:10:31.367257', 'source_status': True, 'number_of_evaluations': 0, 'seeds': None, 'term_limit': 1000}, 'error': None}, {'code': 201, 'resource': 'dataset/5306a78811de5006ba000003', 'location': 'http://bigml.io/andromeda/dataset/5306a78811de5006ba000003', 'object': {'size': 6812, 'code': 201, 'locale': 'en-US', 'updated': '2014-02-21T01:10:32.490681', 'private': True, 'fields_meta': {'count': 0, 'total': 0, 'limit': 1000, 'offset': 0}, 'number_of_predictions': 0, 'category': 0, 'all_fields': True, 'replacements': None, 'source': 'source/5306a782714577095d000000', 'number_of_batchpredictions': 0, 'user_metadata': {}, 'field_types': {'categorical': 0, 'text': 0, 'preferred': 0, 'datetime': 0, 'numeric': 0, 'total': 0}, 'number_of_models': 0, 'number_of_ensembles': 0, 'columns': 0, 'status': {'message': 'The dataset is being processed and will be created soon', 'code': 1}, 'sample_rates': None, 'description': u'', 'tags': [], 'price': 0.0, 'excluded_fields': [], 'ranges': None, 'credits': 0.006496429443359375, 'rows': 0, 'shared': False, 'subscription': True, 'resource': 'dataset/5306a78811de5006ba000003', 'name': "crx_test_2's dataset", 'created': '2014-02-21T01:10:32.490621', 'source_status': True, 'number_of_evaluations': 0, 'seeds': None, 'term_limit': 1000}, 'error': None}), ({'code': 201, 'resource': 'dataset/5306a78b11de5006b0000004', 'location': 'http://bigml.io/andromeda/dataset/5306a78b11de5006b0000004', 'object': {'size': 27155, 'code': 201, 'locale': 'en-US', 'updated': '2014-02-21T01:10:35.278697', 'private': True, 'fields_meta': {'count': 0, 'total': 0, 'limit': 1000, 'offset': 0}, 'number_of_predictions': 0, 'category': 0, 'all_fields': True, 'replacements': None, 'source': 'source/5306a7897145770967000000', 'number_of_batchpredictions': 0, 'user_metadata': {}, 'field_types': {'categorical': 0, 'text': 0, 'preferred': 0, 'datetime': 0, 'numeric': 0, 'total': 0}, 'number_of_models': 0, 'number_of_ensembles': 0, 'columns': 0, 'status': {'message': 'The dataset is being processed and will be created soon', 'code': 1}, 'sample_rates': None, 'description': u'', 'tags': [], 'price': 0.0, 'excluded_fields': [], 'ranges': None, 'credits': 0.02589702606201172, 'rows': 0, 'shared': False, 'subscription': True, 'resource': 'dataset/5306a78b11de5006b0000004', 'name': "crx_train_3's dataset", 'created': '2014-02-21T01:10:35.278655', 'source_status': True, 'number_of_evaluations': 0, 'seeds': None, 'term_limit': 1000}, 'error': None}, {'code': 201, 'resource': 'dataset/5306a78d11de5006ba000006', 'location': 'http://bigml.io/andromeda/dataset/5306a78d11de5006ba000006', 'object': {'size': 6862, 'code': 201, 'locale': 'en-US', 'updated': '2014-02-21T01:10:37.714911', 'private': True, 'fields_meta': {'count': 0, 'total': 0, 'limit': 1000, 'offset': 0}, 'number_of_predictions': 0, 'category': 0, 'all_fields': True, 'replacements': None, 'source': 'source/5306a7897145770959000000', 'number_of_batchpredictions': 0, 'user_metadata': {}, 'field_types': {'categorical': 0, 'text': 0, 'preferred': 0, 'datetime': 0, 'numeric': 0, 'total': 0}, 'number_of_models': 0, 'number_of_ensembles': 0, 'columns': 0, 'status': {'message': 'The dataset is being processed and will be created soon', 'code': 1}, 'sample_rates': None, 'description': u'', 'tags': [], 'price': 0.0, 'excluded_fields': [], 'ranges': None, 'credits': 0.0065441131591796875, 'rows': 0, 'shared': False, 'subscription': True, 'resource': 'dataset/5306a78d11de5006ba000006', 'name': "crx_test_3's dataset", 'created': '2014-02-21T01:10:37.714863', 'source_status': True, 'number_of_evaluations': 0, 'seeds': None, 'term_limit': 1000}, 'error': None}), ({'code': 201, 'resource': 'dataset/5306a7917145770956000000', 'location': 'http://bigml.io/andromeda/dataset/5306a7917145770956000000', 'object': {'size': 27167, 'code': 201, 'locale': 'en-US', 'updated': '2014-02-21T01:10:41.316973', 'private': True, 'fields_meta': {'count': 0, 'total': 0, 'limit': 1000, 'offset': 0}, 'number_of_predictions': 0, 'category': 0, 'all_fields': True, 'replacements': None, 'source': 'source/5306a78f7145770959000003', 'number_of_batchpredictions': 0, 'user_metadata': {}, 'field_types': {'categorical': 0, 'text': 0, 'preferred': 0, 'datetime': 0, 'numeric': 0, 'total': 0}, 'number_of_models': 0, 'number_of_ensembles': 0, 'columns': 0, 'status': {'message': 'The dataset is being processed and will be created soon', 'code': 1}, 'sample_rates': None, 'description': u'', 'tags': [], 'price': 0.0, 'excluded_fields': [], 'ranges': None, 'credits': 0.025908470153808594, 'rows': 0, 'shared': False, 'subscription': True, 'resource': 'dataset/5306a7917145770956000000', 'name': "crx_train_4's dataset", 'created': '2014-02-21T01:10:41.316930', 'source_status': True, 'number_of_evaluations': 0, 'seeds': None, 'term_limit': 1000}, 'error': None}, {'code': 201, 'resource': 'dataset/5306a79211de5006aa000003', 'location': 'http://bigml.io/andromeda/dataset/5306a79211de5006aa000003', 'object': {'size': 6850, 'code': 201, 'locale': 'en-US', 'updated': '2014-02-21T01:10:42.762343', 'private': True, 'fields_meta': {'count': 0, 'total': 0, 'limit': 1000, 'offset': 0}, 'number_of_predictions': 0, 'category': 0, 'all_fields': True, 'replacements': None, 'source': 'source/5306a790714577095a000003', 'number_of_batchpredictions': 0, 'user_metadata': {}, 'field_types': {'categorical': 0, 'text': 0, 'preferred': 0, 'datetime': 0, 'numeric': 0, 'total': 0}, 'number_of_models': 0, 'number_of_ensembles': 0, 'columns': 0, 'status': {'message': 'The dataset is being processed and will be created soon', 'code': 1}, 'sample_rates': None, 'description': u'', 'tags': [], 'price': 0.0, 'excluded_fields': [], 'ranges': None, 'credits': 0.0065326690673828125, 'rows': 0, 'shared': False, 'subscription': True, 'resource': 'dataset/5306a79211de5006aa000003', 'name': "crx_test_4's dataset", 'created': '2014-02-21T01:10:42.762288', 'source_status': True, 'number_of_evaluations': 0, 'seeds': None, 'term_limit': 1000}, 'error': None}), ({'code': 201, 'resource': 'dataset/5306a796714577095a000006', 'location': 'http://bigml.io/andromeda/dataset/5306a796714577095a000006', 'object': {'size': 27184, 'code': 201, 'locale': 'en-US', 'updated': '2014-02-21T01:10:46.235790', 'private': True, 'fields_meta': {'count': 0, 'total': 0, 'limit': 1000, 'offset': 0}, 'number_of_predictions': 0, 'category': 0, 'all_fields': True, 'replacements': None, 'source': 'source/5306a7947145770967000003', 'number_of_batchpredictions': 0, 'user_metadata': {}, 'field_types': {'categorical': 0, 'text': 0, 'preferred': 0, 'datetime': 0, 'numeric': 0, 'total': 0}, 'number_of_models': 0, 'number_of_ensembles': 0, 'columns': 0, 'status': {'message': 'The dataset is being processed and will be created soon', 'code': 1}, 'sample_rates': None, 'description': u'', 'tags': [], 'price': 0.0, 'excluded_fields': [], 'ranges': None, 'credits': 0.0259246826171875, 'rows': 0, 'shared': False, 'subscription': True, 'resource': 'dataset/5306a796714577095a000006', 'name': "crx_train_5's dataset", 'created': '2014-02-21T01:10:46.235754', 'source_status': True, 'number_of_evaluations': 0, 'seeds': None, 'term_limit': 1000}, 'error': None}, {'code': 201, 'resource': 'dataset/5306a79711de5006c1000004', 'location': 'http://bigml.io/andromeda/dataset/5306a79711de5006c1000004', 'object': {'size': 6833, 'code': 201, 'locale': 'en-US', 'updated': '2014-02-21T01:10:47.315224', 'private': True, 'fields_meta': {'count': 0, 'total': 0, 'limit': 1000, 'offset': 0}, 'number_of_predictions': 0, 'category': 0, 'all_fields': True, 'replacements': None, 'source': 'source/5306a7957145770959000006', 'number_of_batchpredictions': 0, 'user_metadata': {}, 'field_types': {'categorical': 0, 'text': 0, 'preferred': 0, 'datetime': 0, 'numeric': 0, 'total': 0}, 'number_of_models': 0, 'number_of_ensembles': 0, 'columns': 0, 'status': {'message': 'The dataset is being processed and will be created soon', 'code': 1}, 'sample_rates': None, 'description': u'', 'tags': [], 'price': 0.0, 'excluded_fields': [], 'ranges': None, 'credits': 0.006516456604003906, 'rows': 0, 'shared': False, 'subscription': True, 'resource': 'dataset/5306a79711de5006c1000004', 'name': "crx_test_5's dataset", 'created': '2014-02-21T01:10:47.315183', 'source_status': True, 'number_of_evaluations': 0, 'seeds': None, 'term_limit': 1000}, 'error': None})]


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
    print 'Max state is: %s\n Accuracy = %f' % (v,fv)
    closed_list.append((v,fv))
    open_list.remove((v,fv))
    if (fv - EPSILON) > best_accuracy:
        best_accuracy = fv
        best_unchanged_count = 0
        print 'new best state'
    else:
        best_unchanged_count += 1

    children = expand_state(v)
    for c in children:
        if (c not in [pair[0] for pair in open_list]
        and c not in [pair[0] for pair in closed_list]):
            input_fields = [id for (i,id) in enumerate(field_ids) if c[i]]
            print 'Evaluating %s' % input_fields
            val = evaluate(input_fields,cv_datasets)
            open_list.append((c,val))

    if best_unchanged_count >= K:
        done = True

print 'CLOSED LIST:'
pprint(closed_list)

print 'OPEN LIST:'
pprint(open_list)
