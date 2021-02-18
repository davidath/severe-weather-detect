import numpy as np
import datetime
import csv
import itertools
import sys
from datetime import datetime as dt
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE as OverSampler
from imblearn.under_sampling import ClusterCentroids as CC
#  from imblearn.under_sampling import RandomUnderSampler as RUS
from sklearn.model_selection import train_test_split

# Logging messages such as loss,loading,etc.

def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(dt.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()

# Util function to save .npz datasets

def make_npz(name, train_data=np.zeros(shape=(1, 1)), test_data=np.zeros(shape=(1, 1)),
             train_labels=np.zeros(shape=(1, 1)), test_labels=np.zeros(shape=(1, 1))):
    np.savez_compressed(name, train_data=train_data, test_data=test_data,
                   test_labels=test_labels, train_labels=train_labels)

# Create Train-Test proof of concept experiment setting

def make_tt(START, END, _csv, ght, TEST_LEN=-1, oversample=False,
        undersample=False, combine=False):
    # Create primary dataset date range
    w_date_range = []
    curr = START
    while True:
        w_date_range.append(curr)
        curr += datetime.timedelta(hours=6)
        if curr == END:
            w_date_range.append(curr)
            break
    w_date_range = np.array(w_date_range)

    # Correlate with CSV dates
    _rows = []
    with open(_csv) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            _rows.append(row)
    _rows = _rows[1:]
    severe = [datetime.datetime.strptime(i[2], '%Y-%m-%d %H:%M:%S') for i in _rows]
    severe = np.unique(severe)

    # Mark as anomalies (0) the dates that are found in the CSV file
    _total = np.ones(shape=(len(w_date_range)))
    for i in severe:
        _total[np.where(w_date_range == i)[0]] = 0

    _total = np.array(_total)
    _total = _total.astype(int)

    if TEST_LEN == -1:
       y = _total
       X = ght
       # check whether to augment anomalous class
       if oversample:
           log('Over-sampling')
           sampler = OverSampler(random_state=1234,
                   sampling_strategy='minority', n_jobs=12)
           X, y = sampler.fit_sample(X, y)
           x_train, x_test, y_train, y_test = train_test_split(X, y,
               test_size=0.33, random_state=1234)
           make_npz('evi_tt_over.npz', train_labels=y_train)
           make_npz('data_tt_over.npz', train_data=x_train,
                   test_data=x_test, train_labels=y_train, test_labels=y_test)
           log('Over-sampling Done!')
       if undersample:
           log('Under-sampling')
           sampler = RUS(random_state=1234, sampling_strategy='majority')
           X, y = sampler.fit_sample(X, y)
           x_train, x_test, y_train, y_test = train_test_split(X, y,
               test_size=0.33, random_state=1234)
           make_npz('evi_tt_under.npz', train_labels=y_train)
           make_npz('data_tt_under.npz', train_data=x_train,
                   test_data=x_test, train_labels=y_train, test_labels=y_test)
           log('Under-sampling Done!')
       if combine:
           log('Combine!')
           sampler = SMOTEENN(random_state=1234, sampling_strategy='all')
           X, y = sampler.fit_sample(X, y)
           x_train, x_test, y_train, y_test = train_test_split(X, y,
               test_size=0.33, random_state=1234)
           make_npz('evi_tt_combine.npz', train_labels=y_train)
           make_npz('data_tt_combine.npz', train_data=x_train,
                   test_data=x_test, train_labels=y_train, test_labels=y_test)
           log('Combine Done!')
       make_npz('evi_tt.npz', train_labels=y_train)
       make_npz('data_tt.npz', train_data=x_train,
               test_data=x_test, train_labels=y_train, test_labels=y_test)
    else:
        # Split dataset in train and test
        _year = [i for i in range(len(w_date_range))]
        test_slic = _year[-TEST_LEN:]
        train_slic = _year[:test_slic[0]]

        # Check whether to augment anomalous class
        if oversample:
            sampler = OverSampler(random_state=1234, sampling_strategy='minority')
            data_tr, evi_tr = sampler.fit_sample(ght[train_slic], _total[train_slic])
            make_npz('evi_tt_over.npz', train_labels=evi_tr)
            make_npz('data_tt_over.npz', train_labels=evi_tr,
                test_labels=_total[test_slic], train_data=data_tr,
                test_data=ght[test_slic])
        else:
            make_npz('evi_tt.npz', train_labels=_total[train_slic])
            make_npz('data_tt.npz', train_labels=_total[train_slic],
                test_labels=_total[test_slic], train_data=ght[train_slic],
                test_data=ght[test_slic])

# Make individual datasets for individual events

def make_indvi(START, END,  _csv, ght, event_name, short=False, short_num=4000):
    np.random.seed(1234)
    # Create primary dataset date range
    w_date_range = []
    curr = START
    while True:
        w_date_range.append(curr)
        curr += datetime.timedelta(hours=6)
        if curr == END:
            w_date_range.append(curr)
            break
    w_date_range = np.array(w_date_range)

    # Correlate with CSV dates
    _rows = []
    with open(_csv) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            _rows.append(row)
    _rows = _rows[1:]
    severe = [datetime.datetime.strptime(i[2], '%Y-%m-%d %H:%M:%S') for i in _rows]
    event = np.array([i[1] for i in _rows])

    # Get unique event indices
    idc = []
    for i in np.unique(event):
        idc.append(np.where(event==i)[0])

    # Assign severe event label
    _total = np.zeros(shape=(len(w_date_range)))
    for c, i in enumerate(np.unique(event)):
        if i == 'Windstorm':
            _total[idc[c]] = 1
        elif i == 'Flood':
            _total[idc[c]] = 2
        elif i == 'Hail':
            _total[idc[c]] = 3
        elif i == 'Tornado':
            _total[idc[c]] = 4

    _total = _total.astype(int)
    # Short version
    # ------------------------------
    if short:
        idx = []
        for i in np.unique(_total):
            if i == 0:
                zero = np.where(_total == i)[0]
                p = np.random.permutation(zero.size)
                zero = zero[p]
                idx.append(zero[:short_num])
            else:
                idx.append(np.where(_total == i)[0])

        idx = list(itertools.chain(*idx))
        idx = np.array(idx)
        _total = _total[idx]
        _total[np.where(_total != 0)] = 1
    # ------------------------------

    # Select event type
    if event_name == 'Windstorm':
       total_slic = np.concatenate((np.where(_total == 0)[0], np.where(_total == 1)[0]))
    elif event_name == 'Flood':
       total_slic = np.concatenate((np.where(_total == 0)[0], np.where(_total == 2)[0]))
    elif event_name == 'Hail':
       total_slic = np.concatenate((np.where(_total == 0)[0], np.where(_total == 3)[0]))
    elif event_name == 'Tornado':
       total_slic = np.concatenate((np.where(_total == 0)[0], np.where(_total == 4)[0]))
    else:
       total_slic = range(0, len(_total))
    y = _total[total_slic]
    X = ght[total_slic]
    x_train, x_test, y_train, y_test = train_test_split(X, y,
       test_size=0.33, random_state=1234)

    make_npz('evi_'+event_name+'.npz',  train_labels=y_train)
    make_npz('data_'+event_name+'.npz', train_data=x_train,
               test_data=x_test, train_labels=y_train, test_labels=y_test)

# Make 1 event evidence and rest clustering task

def make_oneall(START, END,  _csv, ght, event_name, short=False, short_num=4000):
    np.random.seed(1234)
    # Create primary dataset date range
    w_date_range = []
    curr = START
    while True:
        w_date_range.append(curr)
        curr += datetime.timedelta(hours=6)
        if curr == END:
            w_date_range.append(curr)
            break
    w_date_range = np.array(w_date_range)

    # Correlate with CSV dates
    _rows = []
    with open(_csv) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            _rows.append(row)
    _rows = _rows[1:]
    severe = [datetime.datetime.strptime(i[2], '%Y-%m-%d %H:%M:%S') for i in _rows]
    event = np.array([i[1] for i in _rows])

    # Get unique event indices
    idc = []
    for i in np.unique(event):
        idc.append(np.where(event==i)[0])

    # Assign severe event label
    _total = np.zeros(shape=(len(w_date_range)))
    for c, i in enumerate(np.unique(event)):
        if i == 'Windstorm':
            _total[idc[c]] = 1
        elif i == 'Flood':
            _total[idc[c]] = 2
        elif i == 'Hail':
            _total[idc[c]] = 3
        elif i == 'Tornado':
            _total[idc[c]] = 4

    _total = _total.astype(int)

    # Short version
    # ------------------------------
    if short:
        idx = []
        for i in np.unique(_total):
            if i == 0:
                zero = np.where(_total == i)[0]
                p = np.random.permutation(zero.size)
                zero = zero[p]
                idx.append(zero[:short_num])
            else:
                idx.append(np.where(_total == i)[0])

        idx = list(itertools.chain(*idx))
        idx = np.array(idx)
        _total = _total[idx]
    # ------------------------------

    # Select event type
    if event_name == 'Windstorm':
       train_slic = np.where(_total == 1)[0]
       other_slic = np.where(_total != 1)[0]
    elif event_name == 'Flood':
       train_slic = np.where(_total == 2)[0]
       other_slic = np.where(_total != 2)[0]
    elif event_name == 'Hail':
       train_slic = np.where(_total == 3)[0]
       other_slic = np.where(_total != 3)[0]
    elif event_name == 'Tornado':
       train_slic = np.where(_total == 4)[0]
       other_slic = np.where(_total != 4)[0]
    else:
        print ('Event name not found!')
        exit()

    X = ght[range(0, len(_total))]
    y = _total[range(0, len(_total))]
    x_train, x_test, y_train, y_test = train_test_split(X, y,
       test_size=0.33, random_state=1234)

    evi_data = X[np.concatenate((train_slic, other_slic))]
    evi_lab = _total[np.concatenate((train_slic, other_slic))]
    evi_lab[:len(train_slic)] = 0
    evi_lab[len(train_slic):len(train_slic)+len(other_slic)] = 1

    perm = np.random.permutation(len(evi_data))
    evi_data = evi_data[perm]
    evi_lab = evi_lab[perm]

    make_npz('evi_1all_'+event_name+'.npz',  train_data=evi_data, train_labels=evi_lab)
    make_npz('data_1all_'+event_name+'.npz', train_data=x_train,
           test_data=x_test, train_labels=y_train, test_labels=y_test)

# Make 1 event evidence and rest clustering task

def make_allone(START, END,  _csv, ght, event_name, event_name2, short=False, short_num=4000):
    np.random.seed(1234)
    # Create primary dataset date range
    w_date_range = []
    curr = START
    while True:
        w_date_range.append(curr)
        curr += datetime.timedelta(hours=6)
        if curr == END:
            w_date_range.append(curr)
            break
    w_date_range = np.array(w_date_range)

    # Correlate with CSV dates
    _rows = []
    with open(_csv) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            _rows.append(row)
    _rows = _rows[1:]
    severe = [datetime.datetime.strptime(i[2], '%Y-%m-%d %H:%M:%S') for i in _rows]
    event = np.array([i[1] for i in _rows])

    # Get unique event indices
    idc = []
    for i in np.unique(event):
        idc.append(np.where(event==i)[0])

    # Assign severe event label
    _total = np.zeros(shape=(len(w_date_range)))
    for c, i in enumerate(np.unique(event)):
        if i == 'Windstorm':
            _total[idc[c]] = 1
        elif i == 'Flood':
            _total[idc[c]] = 2
        elif i == 'Hail':
            _total[idc[c]] = 3
        elif i == 'Tornado':
            _total[idc[c]] = 4

    _total = _total.astype(int)

    # Short version
    # ------------------------------
    if short:
        idx = []
        for i in np.unique(_total):
            if i == 0:
                zero = np.where(_total == i)[0]
                p = np.random.permutation(zero.size)
                zero = zero[p]
                idx.append(zero[:short_num])
            else:
                idx.append(np.where(_total == i)[0])

        idx = list(itertools.chain(*idx))
        idx = np.array(idx)
        _total = _total[idx]
    # ------------------------------

    # Select event type
    if event_name == 'Windstorm':
       train_slic = np.where(_total == 1)[0]
       #  other_slic = np.where(_total != 1)[0]
       if event_name2 == 'Flood':
           zero = np.where(_total == 0)[0]
           two = np.where(_total == 2)[0]
       elif event_name2 == 'Tornado':
           zero = np.where(_total == 0)[0]
           two = np.where(_total == 4)[0]
       other_slic = np.concatenate((zero, two))
    elif event_name == 'Flood':
       train_slic = np.where(_total == 2)[0]
       #  other_slic = np.where(_total != 2)[0]
       if event_name2 == 'Windstorm':
           zero = np.where(_total == 0)[0]
           two = np.where(_total == 1)[0]
       elif event_name2 == 'Tornado':
           zero = np.where(_total == 0)[0]
           two = np.where(_total == 4)[0]
       other_slic = np.concatenate((zero, two))
    elif event_name == 'Hail':
       train_slic = np.where(_total == 3)[0]
       other_slic = np.where(_total != 3)[0]
    elif event_name == 'Tornado':
       train_slic = np.where(_total == 4)[0]
       #  other_slic = np.where(_total != 4)[0]
       if event_name2 == 'Windstorm':
           zero = np.where(_total == 0)[0]
           two = np.where(_total == 1)[0]
       elif event_name2 == 'Flood':
           zero = np.where(_total == 0)[0]
           two = np.where(_total == 2)[0]
       other_slic = np.concatenate((zero, two))
    else:
        print ('Event name not found!')
        exit()

    X = ght[np.concatenate((train_slic, other_slic))]
    y = _total[np.concatenate((train_slic, other_slic))]
    y[:len(train_slic)] = 0
    y[len(train_slic):len(train_slic)+len(other_slic)] = 1
    p = np.random.permutation(len(X))
    print 'Data:'
    for c,i in enumerate(np.unique(y)):
        print i, len(np.where(y == i)[0])

    make_npz('data_all1_'+event_name+'_'+event_name2+'.npz', train_data=X[p], train_labels=y[p])

    #  idx = remove_labs(y, 1, len(np.where(y == 0)[0]))
    #  y = y[idx]
    #  X = X[idx]

    #  sampler = OverSampler(random_state=1234,
           #  sampling_strategy='minority', n_jobs=12)
    #  X, y = sampler.fit_sample(X, y)

    evi_data = ght[np.concatenate((train_slic, other_slic))]
    evi_lab = _total[np.concatenate((train_slic, other_slic))]
   
    if event_name == 'Windstorm' and event_name2 == 'Flood':
        evi_lab[np.where(evi_lab == 0)[0]] = 0
        evi_lab[np.where(evi_lab == 1)[0]] = 0
        evi_lab[np.where(evi_lab == 2)[0]] = 1
    elif event_name == 'Windstorm' and event_name2 == 'Tornado':
        evi_lab[np.where(evi_lab == 0)[0]] = 0
        evi_lab[np.where(evi_lab == 1)[0]] = 0
        evi_lab[np.where(evi_lab == 4)[0]] = 1
    elif event_name == 'Flood' and event_name2 == 'Windstorm':
        evi_lab[np.where(evi_lab == 0)[0]] = 0
        evi_lab[np.where(evi_lab == 2)[0]] = 0
        evi_lab[np.where(evi_lab == 1)[0]] = 1
    elif event_name == 'Flood' and event_name2 == 'Tornado':
        evi_lab[np.where(evi_lab == 0)[0]] = 0
        evi_lab[np.where(evi_lab == 2)[0]] = 0
        evi_lab[np.where(evi_lab == 4)[0]] = 1
    elif event_name == 'Tornado' and event_name2 == 'Windstorm':
        evi_lab[np.where(evi_lab == 0)[0]] = 0
        evi_lab[np.where(evi_lab == 4)[0]] = 0
        evi_lab[np.where(evi_lab == 1)[0]] = 1
    elif event_name == 'Tornado' and event_name2 == 'Flood':
        evi_lab[np.where(evi_lab == 0)[0]] = 0
        evi_lab[np.where(evi_lab == 4)[0]] = 0
        evi_lab[np.where(evi_lab == 2)[0]] = 1

    #  for c,i in enumerate(np.unique(evi_lab)):
        #  evi_lab[np.where(evi_lab == i)[0]] = c

    #  idx = remove_labs(evi_lab, 0, int(len(np.where(_total == 0)[0]) * 0.3))
    #  idx = remove_labs(evi_lab, 0, 0)
    #  evi_lab = evi_lab[idx]
    #  evi_data = evi_data[idx]


    #  idx = np.where(evi_lab == 1)[0]
    #  other_idx = np.where(evi_lab != 1)[0]
    #  temp_lab = np.copy(evi_lab)
    #  temp_lab[idx] = 0
    #  temp_lab[other_idx] = 1
    #  make_npz('evi_all1_'+event_name+'.npz', train_data=evi_data,
            #  train_labels = temp_lab)
     
    print 'Evidence:'
    for c,i in enumerate(np.unique(evi_lab)):
        print i, len(np.where(evi_lab == i)[0])
    #  p = np.random.permutation(len(evi_lab))

    #  sampler = OverSampler(random_state=1234,
           #  sampling_strategy='all', n_jobs=12)
    #  evi_data, evi_lab = sampler.fit_sample(evi_data, evi_lab)

    make_npz('evi_all1_'+event_name+'_'+event_name2+'.npz',  train_data=evi_data[p],
            train_labels=evi_lab[p])


def remove_labs(labels, rmlab, keep_num):
    idx = []
    for i in np.unique(labels):
        if i == rmlab:
            zero = np.where(labels == i)[0]
            p = np.random.permutation(zero.size)
            zero = zero[p]
            if keep_num == 0:
                continue
            else:
                idx.append(zero[:keep_num])
        else:
            idx.append(np.where(labels == i)[0])

    idx = list(itertools.chain(*idx))
    idx = np.array(idx)
    return idx

def make_dayframes(dataset, frames):
    if dataset.shape[0] % frames == 0:
        dataset = np.array([dataset[i:i+frames] for i in range(0, len(dataset), frames)])
        dataset = dataset.reshape(dataset.shape[0], frames, dataset.shape[2])
    else:
        div = int(dataset.shape[0] / frames)
        dataset = np.array([dataset[i:i+frames] for i in range(0, div*frames, frames)])
        dataset = dataset.reshape(dataset.shape[0], frames, dataset.shape[2])
    return dataset

def make_dayframes_labels(dataset, frames):
    if dataset.shape[0] % frames == 0:
        dataset = [int(np.average(dataset[i:i+frames])) for i in range(0, len(dataset), frames)]
        dataset = np.array(dataset)
    else:
        div = int(dataset.shape[0] / frames)
        dataset = [int(np.average(dataset[i:i+frames])) for i in range(0, div * frames, frames)]
        dataset = np.array(dataset)
    return dataset

# Create Train-Test proof of concept experiment setting (With temporality)

def make_tt_tempo(START, END, _csv, ght, TEST_LEN=-1, frames=4, oversample=False,
        undersample=False, combine=False):
    np.random.seed(1234)
    # Create primary dataset date range
    w_date_range = []
    curr = START
    while True:
        w_date_range.append(curr)
        curr += datetime.timedelta(hours=6)
        if curr == END:
            w_date_range.append(curr)
            break
    w_date_range = np.array(w_date_range)

    # Correlate with CSV dates
    _rows = []
    with open(_csv) as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            _rows.append(row)
    _rows = _rows[1:]
    severe = [datetime.datetime.strptime(i[2], '%Y-%m-%d %H:%M:%S') for i in _rows]
    severe = np.unique(severe)

    # Mark as anomalies (0) the dates that are found in the CSV file
    _total = np.ones(shape=(len(w_date_range)))
    for i in severe:
        _total[np.where(w_date_range == i)[0]] = 0

    _total = np.array(_total)
    _total = _total.astype(int)

    X = make_dayframes(ght, frames)
    y = make_dayframes_labels(_total, frames)
    make_npz('data_tt_tempo', train_data=X, train_labels=y)

    if undersample:
        SHORT_NUM = 1000
        idx = []
        for i in np.unique(y):
            if i == 1:
                zero = np.where(y == i)[0]
                p = np.random.permutation(zero.size)
                zero = zero[p]
                idx.append(zero[:SHORT_NUM])
            else:
                idx.append(np.where(y == i)[0])

        idx = list(itertools.chain(*idx))
        idx = np.array(idx)
        y = y[idx]
        X = X[idx]
        x_train, x_test, y_train, y_test = train_test_split(X, y,
           test_size=0.33, random_state=1234)
        make_npz('evi_tt_tempo_under.npz', train_labels=y_train)
        make_npz('data_tt_tempo_under.npz', train_data=x_train,
               test_data=x_test, train_labels=y_train, test_labels=y_test)

#Multiply samples by repeating N times the samples using small uniform noise

def noise(_in, channels, repeat_num):
    noisy = []
    for j in range(num):
        sample = []
        div = _in.shape[1]/channels
        for k in range(repeat_num):
            rng = np.random.RandomState().uniform(-0.1, 0.1, size=div)
            noise = np.add(_in[i][k*div:(k+1)*div], np.multiply(_in[i][k*div:(k+1)*div], rng))
            sample.append(noise)
        sample = list(itertools.chain(*sample))
        noisy.append(np.array(sample))
    noisy = np.array(noisy)
    return noisy

# Same as noise, manual version to confirm frame order for temporal use

def noise_confirm_order(_in, channels):
    noisy = []
    # Original
    for i in range(len(_in)):
        noisy.append(_in[i])
    # 1
    for i in range(len(_in)):
        sample = []
        div = _in.shape[1]/channels
        for k in range(channels):
            rng = np.random.RandomState().uniform(-0.1, 0.1, size=div)
            noise = np.add(_in[i][k*div:(k+1)*div], np.multiply(_in[i][k*div:(k+1)*div], rng))
            sample.append(noise)
        sample = list(itertools.chain(*sample))
        noisy.append(np.array(sample))
    # 2
    for i in range(len(_in)):
        sample = []
        div = _in.shape[1]/channels
        for k in range(channels):
            rng = np.random.RandomState().uniform(-0.1, 0.1, size=div)
            noise = np.add(_in[i][k*div:(k+1)*div], np.multiply(_in[i][k*div:(k+1)*div], rng))
            sample.append(noise)
        sample = list(itertools.chain(*sample))
        noisy.append(np.array(sample))
    # 3
    for i in range(len(_in)):
        sample = []
        div = _in.shape[1]/channels
        for k in range(channels):
            rng = np.random.RandomState().uniform(-0.1, 0.1, size=div)
            noise = np.add(_in[i][k*div:(k+1)*div], np.multiply(_in[i][k*div:(k+1)*div], rng))
            sample.append(noise)
        sample = list(itertools.chain(*sample))
        noisy.append(np.array(sample))
    # 4
    for i in range(len(_in)):
        sample = []
        div = _in.shape[1]/channels
        for k in range(channels):
            rng = np.random.RandomState().uniform(-0.1, 0.1, size=div)
            noise = np.add(_in[i][k*div:(k+1)*div], np.multiply(_in[i][k*div:(k+1)*div], rng))
            sample.append(noise)
        sample = list(itertools.chain(*sample))
        noisy.append(np.array(sample))
    noisy = np.array(noisy)
    return noisy
