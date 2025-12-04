#!/usr/bin/env python3

import argparse
import os
import sys

DEFAULT_DATASETS_DIR = "source-datasets"

def get_files_in_dir(dir: str, recursive: bool = False):
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
                
if __name__ == "__main__":
    # parse arguments
    argparser = argparse.ArgumentParser()
    subset_size_parser = argparser.add_mutually_exclusive_group(required=True)
    subset_size_parser.add_argument("--examples", "-E", type=int, help="Number of examples in the subset")
    subset_size_parser.add_argument("--percent", "-P", type=float, help="Percent of source dataset(s) to include in subset")
    argparser.add_argument("-r", action="store_true", dest="recursive", help="Add all files in directories specified recursively")
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
    
    