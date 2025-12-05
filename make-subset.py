#!/usr/bin/env python3

import argparse
import os
import sys
import random
import typing
import math
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_DATASETS_DIR = "source-datasets"
kSEED = 0

random.seed(kSEED)

def get_files_in_dir(dir: str, recursive: bool = False) -> typing.List[str]:
    """
    Return a list of all filenames in dir. Filenames returned are relative to the directory in which the script was run.
    
    :param dir: The name of the directory
    :type dir: str
    :param recursive: Return the files in every directory in dir as well
    :type recursive: bool
    :rtype: List[str]
    """
    files = []
    dir_files = os.listdir(dir)
    # add basename to all filenames in dir_files
    for i in range(len(dir_files)):
        dir_files[i] = dir + os.sep + dir_files[i]

    for file in dir_files:
        if os.path.isdir(file):
            if recursive:  # add all files in directory if recursive is true
                files.extend(get_files_in_dir(file, recursive))
        elif os.path.isfile(file):  # add file to list if file exists
            files.append(file)
    return files

def num_positive_examples(df: pd.DataFrame) -> int:
    """
    Return the number of positive examples in df. df must have a column named "label".
    
    :param df: A Pandas DataFrame with a column named "label"
    :type df: pd.DataFrame
    :return: The number of positive examples in df
    :rtype: int
    """
    return df[df.label == 1].shape[0]

def num_negative_examples(df: pd.DataFrame) -> int:
    """
    Return the number of negative examples in df. df must have a column named "label".
    
    :param df: A Pandas DataFrame with a column named "label"
    :type df: pd.DataFrame
    :return: The number of negative examples in df
    :rtype: int
    """
    return df[df.label == 0].shape[0]

def percent_positive_examples(df: pd.DataFrame) -> float:
    """
    Return a float representing the percent of positive examples in df. df must have a column named "label".
    
    :param df: A Pandas Dataframe with a column named "label"
    :type df: pd.DataFrame
    :return: The percent of positive examples in df.
    :rtype: float
    """
    total_examples = len(df.index)
    return float(num_positive_examples(df)) / total_examples
                
if __name__ == "__main__":
    # parse arguments
    argparser = argparse.ArgumentParser()
    subset_size_parser = argparser.add_mutually_exclusive_group(required=True)
    subset_size_parser.add_argument("--examples", "-E", type=int, help="Number of examples in the subset")
    subset_size_parser.add_argument("--percent", "-P", type=float, help="Percent of source dataset(s) to include in subset")
    argparser.add_argument("-r", action="store_true", dest="recursive", help="Add all files in directories specified recursively")
    argparser.add_argument("--outfile", "-O", type=str, required=True, help="File to write resulting subset to")
    argparser.add_argument("files", nargs="*", help="Files and/or directories to use")
    args = argparser.parse_args()

    # add datasets to args.files if no files supplied
    if len(args.files) == 0:
        datasets = os.listdir(DEFAULT_DATASETS_DIR)
        datasets.remove("Vectorized")  # remove vectorized datasets from list
        for i in range(len(datasets)):
            datasets[i] = DEFAULT_DATASETS_DIR + os.sep + datasets[i]
        args.files.extend(datasets)

    # create list of files
    files = []
    for file in args.files:
        if os.path.isdir(file):  # add all files in file to list if file is a directory
            files.extend(get_files_in_dir(file, args.recursive))
        elif os.path.isfile(file):  # add file to list if file exists
            files.append(file)
    
    # ensure at least one file is in list
    if len(files) == 0:
        sys.stderr.write("Error: files list must contain at least one real file. Maybe you're having file path issues?\n")
        exit(1)

    # get total length of datasets
    full_dataset_length = 0
    full_dataset_positive_examples = 0
    for file in files:
        dataset = pd.read_csv(file, lineterminator='\n')
        full_dataset_length += len(dataset.index)
        full_dataset_positive_examples += num_positive_examples(dataset)

    # create stratified (representative) splitter object
    if args.examples:  # transform number of examples into percent
        subset_percent = float(args.examples) / full_dataset_length
    else:
        subset_percent = args.percent
        print(f"subset_percent is {subset_percent}") #DEBUG
    sss = StratifiedShuffleSplit(
        n_splits=1,                 # number of splits (subsets) needed
        train_size=subset_percent,  # percent of examples in subset. train_size is a percent if it's a float
        test_size=1,
        random_state=kSEED)         # random seed for consistency
    
    # generate subset
    subset = pd.DataFrame()
    for file in files:
        dataset = pd.read_csv(file, lineterminator='\n')
        # get training data split
        train_idx, _ = next(sss.split(dataset, dataset["label"]))
        temp_subset = dataset.loc[train_idx].reset_index(drop=True)
        # concatenate body and label columns from temp_subset to subset
        subset = pd.concat([subset, temp_subset[["body", "label"]]], ignore_index=True)

    # if examples was specified, ensure subset has the correct amount of examples
    if args.examples and len(subset.index) < args.examples:  # subset has less examples than required
        num_examples_needed = args.examples - len(subset.index)
        samples_per_file = int(math.ceil(float(num_examples_needed) / len(files)))  # ensure there will be enough samples to cover num_examples_needed

        # sample datasets for additional examples
        i = 0
        for file in files:
            dataset = pd.read_csv(file, lineterminator='\n')
            j = 0
            while j < samples_per_file and i < num_examples_needed:
                random_row = dataset.sample(n=1, random_state=kSEED)
                subset = pd.concat([subset, random_row[["body", "label"]]], ignore_index=True)  # add random row to subset
                j += 1
                i += 1
            if i >= num_examples_needed:
                break
    elif args.examples and len(subset.index) > args.examples:  # subset has more examples than required
        num_excess_examples = len(subset.index) - args.examples
        # Remove num_excess_examples random rows from the subset
        subset = subset.sample(n=args.examples, random_state=kSEED).reset_index(drop=True)
    
    print(subset) #DEBUG

        

    # df = pd.read_csv(files[0])
    # # get training data split from 
    # train_idx, _ = next(sss.split(df, df['label']))
    # subset = df.loc[train_idx].reset_index(drop=True)
    # print(subset)
    # print(f"subset has {(percent_positive_examples(subset) * 100):.2f}% positive examples")

    # create outfile
    outfile = os.open(args.outfile, flags=(os.O_WRONLY | os.O_CREAT))

    # close outfile
    os.close(outfile)