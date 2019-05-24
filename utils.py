import os
import glob
import numpy as np
np.random.seed(0)
import random as rn
rn.seed(0)
import csv
import re
import sys
import shutil
import pandas as pd
from tqdm import tqdm
import math

def setConfigByType(cfg, type_, name=None):
    """
    Setting configuration for training
    :param cfg: the config dict loaded for training
    :param type_: type of model
    :param name: name of natural community
    :return: cfg
    """
    assert (type_ in ["binary", "multi", "rank"]), 'Type not exists'
    cfg.type = type_
    cfg.cls_name = cfg.classes[type_]
    cfg.cls_num = len(cfg.cls_name)
    cfg.cls_dict = dict([(cfg.cls_name[i], i) for i in range(cfg.cls_num)])
    if type_ == 'rank':
        assert(name is not None), 'Natural community name is not specified'
    cfg.nc_name = name
    cfg.solution_name = type_ + ("_"+name if name else "")
    return cfg


def prepareDataFrame(df, type, name=None, train_val_split=0.2):
    """
    Prepare data frame and train validation split. We should split the data first and then do oversampling for only train set
    :param type: type of data to prepare,  'binary': separate NC and background, 'multi': separate different NCs, 'rank': rank classification( if 'rank' is given, name of NC should be given)
    :param name: name of the specified natural community
    :param train_val_split: train validation split ratio
    :return: dataframe for train and validation
    """
    assert(type in ["binary", "multi", "rank"]), 'Type not exists'
    # Permut instances per each specie
    permutDf = pd.DataFrame()
    uqArr = np.unique(df['specie'])
    for uq in uqArr:
        df2 = df.loc[df.loc[df['specie'] == uq].index]
        df2 = df2.reindex(np.random.permutation(df2.index))
        permutDf = permutDf.append(df2, ignore_index=True)
    df = permutDf

    if type == 'binary':
        # Specify all species names except NEGATIVES as POSITIVES
        df.at[df.loc[df['specie'] != 'NEGATIVES'].index, 'specie'] = 'POSITIVES'
    elif type == 'multi':
        # Drop out NEGATIVES
        df = df.drop(df.loc[df['specie'] == 'NEGATIVES'].index)
    elif type == 'rank':
        assert (name is not None),  'Natural community name is not specified'
        # Keep rows by name
        df = df.drop(df.loc[df['specie'] != name].index)
        ranks = df.eo_rank.unique()
        for rank in ranks:
            df.at[df.loc[df['eo_rank'] == rank].index, 'specie'] = rank
    df = df.reset_index()

    # create train, valid dataframe
    train = pd.DataFrame(columns=list(df.columns.values))
    test = pd.DataFrame(columns=list(df.columns.values))
    specieNbs = df['specie'].value_counts()
    for x in specieNbs.items():
        idx = np.array(df.index[df.specie == x[0]].tolist())
        val_split = int(len(idx)*train_val_split)
        test = test.append(df.iloc[idx[:val_split]])
        train = train.append(df.iloc[idx[val_split:]])

    # random oversampling
    specieNbs = train['specie'].value_counts()
    maxSpecieNb = max(specieNbs)
    for x in tqdm(specieNbs.items()):
        needAdd = maxSpecieNb - x[1]
        indices = np.array(train.loc[train['specie'] == x[0]].index.tolist())
        sample_idx = indices[np.random.randint(0, len(indices), size=needAdd)]
        train = train.append(train.iloc[sample_idx], ignore_index=True)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test

def setCommunities(cfg):
    """
    Setting natural communities list in configuration
    :param cfg: cfg
    :return: cfg
    """
    community = os.listdir(cfg.data_path)
    community.remove('NEGATIVES')
    cfg.classes['multi'] = community
    cfg.cls_num = len(community)
    cfg.cls_dict = dict([(community[i], i) for i in range(cfg.cls_num)])
    return cfg

def generateCsv():
    """
    Create csv file for patches
    :param data_path: path to the patch data
    :return: None
    """
    data_path='/home/ubuntu/Projects/tnc_ai/thor/Data/patch/87_withNeg_size224'
    community_names = os.listdir(data_path)
    with open("/home/ubuntu/Projects/tnc_ai/peter/model/EO_rank_data.csv", mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(["filename", "specie", "eo_rank"])
        for community_name in tqdm(community_names):
            community_path = os.path.join(data_path, community_name)
            ranks = os.listdir(community_path)
            for rank in ranks:
                rank_path = os.path.join(community_path, rank, "pos")
                imgs = glob.glob(os.path.join(rank_path, "*.png"))
                for img_path in imgs:
                    csv_writer.writerow([img_path, community_name, rank])

def getCheckpoints(dirName, fileNameStarter):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            if entry.startswith('.'):
                print('Checkpoint ignored in folder {0}'.format(fullPath))
            else:
                allFiles = allFiles + getCheckpoints(fullPath, fileNameStarter)
        else:
            if entry.endswith('.h5') and entry.startswith(fileNameStarter):
                allFiles.append(fullPath)
    return allFiles

# convert string to multi-label binary class vector, zeros for negative samples
def multiLabelBinarier(labels, cls_num, rank_num):
    n = labels.shape[0]
    y1 = np.zeros([n, cls_num])
    y2 = np.zeros([n, rank_num])
    y1[range(n), labels[:,0]] = 1
    y2[range(n), labels[:,1]] = 1
    return np.hstack([y1, y2])

# wrapper generator for multilabel
def wrapperGenerator(generator, cls_num, rank_num):
    while True:
        X, y = next(generator)
        y = multiLabelBinarier(y, cls_num, rank_num)
        yield(X, y)

def parseCheckpointFilename(line):
    mask = re.compile(r"val_loss([-]?\d+(?:\.\d+)?)-val_acc([-]?\d+(?:\.\d+)?)")
    # mask = re.compile(r"val_loss([-]?[0-9.]+)")
    # mask = re.compile(r"val_loss([-]?[0-9.]+)-val_mean_iou([-]?[0-9]+[.]?[0-9]+)")
    res = mask.search(os.path.basename(line))
    if res:
        return (float(item) for item in res.groups(0))
    else:
        raise ValueError("Wrong checkpoint filename format: {}".format(line))

def recreateFolder(fname):
    if os.path.isdir(fname)==True:
        shutil.rmtree(fname)
    os.makedirs(fname)

def getBestCheckpoint(checkpoints):
    best_val_param = 10.0
    best_index = -1
    for ind, file in enumerate(checkpoints):
        try:
            val_loss, val_acc = parseCheckpointFilename(file)
            if val_acc > best_val_param or best_index < 0:
                best_val_param = val_acc
                best_index = ind
        except Exception as err:
            err_str = str(err)
            print('ERROR:', err_str)
    return best_index


def FindBestCheckpoint(rootdir, fileNameStarter):
    checkpoints = getCheckpoints(rootdir, fileNameStarter)
    best_checkpoint_index = getBestCheckpoint(checkpoints)
    if best_checkpoint_index < 0:
        return ("")
    val_loss, val_acc = parseCheckpointFilename(checkpoints[best_checkpoint_index])
    print('Found best checkpoint: {}'.format(checkpoints[best_checkpoint_index]))
    print('val_loss: {}, val_acc: {}'.format(val_loss, val_acc))
    return checkpoints[best_checkpoint_index]

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def getTiledBoxx(img_shape, tile_size, offset):
    aw0 = []
    aw1 = []
    ah0 = []
    ah1 = []
    for i in range(int(math.ceil(1 + (img_shape[0] - tile_size[1])/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(1 + (img_shape[1] - tile_size[0])/(offset[0] * 1.0)))):
            h1 = min(offset[1]*i+tile_size[1], img_shape[0])
            h0 = max(0, h1 - tile_size[1])
            w1 = min(offset[0]*j+tile_size[0], img_shape[1])
            w0 = max(0, w1 - tile_size[0])
            aw0.append(w0)
            aw1.append(w1)
            ah0.append(h0)
            ah1.append(h1)
    return aw0,aw1,ah0,ah1

if __name__=="__main__":
    generateCsv()
