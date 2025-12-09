#!/usr/bin/env python3

import argparse
import os
import sys
import random
import typing
import math
import pandas as pd
#from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_DATASETS_DIR = "source-datasets"
VERIFICATION_CLOSENESS = 0.02
PRECISION = 4
kSEED = 0
random.seed(kSEED)
verbose: bool = False
silent: bool = False

def print_verbose(*values: object):
    """
    Print the given message or object if the verbose flag is set.
    
    :param values: The message or object to be printed
    :type values: object
    """
    if verbose:
        print(*values, flush=True)

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
                print_verbose(f"Found directory {file} in recursive mode, adding all its files to file list.")  # VERBOSE
                files.extend(get_files_in_dir(file, recursive))
        elif os.path.isfile(file):  # add file to list if file exists
            print_verbose(f"Found file {file}, adding it to file list.")  # VERBOSE
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

def proportion_positive_examples(df: pd.DataFrame) -> float:
    """
    Return a float representing the proportion of positive examples in df. df must have a column named "label".
    
    :param df: A Pandas Dataframe with a column named "label"
    :type df: pd.DataFrame
    :return: The proportion of positive examples in df.
    :rtype: float
    """
    total_examples = len(df.index)
    return float(num_positive_examples(df)) / total_examples
                
if __name__ == "__main__":
    # parse arguments
    argparser = argparse.ArgumentParser()
    subset_size_parser = argparser.add_mutually_exclusive_group(required=True)
    subset_size_parser.add_argument("--examples", "-E", type=int, help="Number of examples in the subset")
    subset_size_parser.add_argument("--proportion", "-P", type=float, help="Proportion of source dataset(s) to include in subset")
    argparser.add_argument("--outfile", "-O", type=str, required=True, help="File to write resulting subset to")
    argparser.add_argument("-r", action="store_true", dest="recursive", help="Add all files in directories specified recursively")
    argparser.add_argument("--verify", action="store_true", help="Verify that the file written to has the correct number of examples")
    argparser.add_argument("--precision", type=int, required=False, help="The precision to print floating point numbers to the terminal with.")
    verboseness_parser = argparser.add_mutually_exclusive_group(required=False)
    verboseness_parser.add_argument("--verbose", action="store_true", help="Print additional messages while running")
    verboseness_parser.add_argument("--silent", action="store_true", help="Print no messages while running")
    argparser.add_argument("files", nargs="*", help="Files and/or directories to use")
    args = argparser.parse_args()

    if args.verbose:
        verbose = True
    elif args.silent:
        silent = True
    
    if args.precision:
        PRECISION = args.precision

    # add datasets to args.files if no files supplied
    if len(args.files) == 0:
        print_verbose(f"No files supplied, using default files from {DEFAULT_DATASETS_DIR} directory.")  # VERBOSE
        datasets = os.listdir(DEFAULT_DATASETS_DIR)
        datasets.remove("Vectorized")  # remove vectorized datasets from list
        for i in range(len(datasets)):
            datasets[i] = DEFAULT_DATASETS_DIR + os.sep + datasets[i]
        args.files.extend(datasets)

    # create list of files
    files = []
    for file in args.files:
        if os.path.isdir(file):  # add all files in file to list if file is a directory
            print_verbose(f"Found directory {file}, adding all its files to file list.")  # VERBOSE
            files.extend(get_files_in_dir(file, args.recursive))
        elif os.path.isfile(file):  # add file to list if file exists
            print_verbose(f"Found file {file}, adding it to file list.")  # VERBOSE
            files.append(file)
    
    # ensure at least one file is in list
    if len(files) == 0:
        sys.stderr.write("Error: Files list must contain at least one real file. Maybe you're having file path issues?\n")
        exit(1)

    # get total length of datasets
    """this could be much more efficient if the csv was parsed manually, but oh well"""
    print_verbose("Calculating total length of datasets in file list.")  # VERBOSE
    full_dataset_length = 0
    full_dataset_positive_examples = 0
    for file in files:
        print_verbose(f"Reading file {file} to determine length.")  # VERBOSE
        dataset = pd.read_csv(file, lineterminator='\n')
        full_dataset_length += len(dataset.index)
        full_dataset_positive_examples += num_positive_examples(dataset)
        print_verbose(f"Found {full_dataset_length} examples in file {file}, {full_dataset_positive_examples} of which were positive.")  # VERBOSE
    full_dataset_positive_proportion = float(full_dataset_positive_examples) / full_dataset_length

    # print length
    if not silent: print(f"Finished reading files for dataset length, found {full_dataset_length} examples.", flush=True)
    
    # set proportion
    if args.examples:  # calculate proportion if the examples argument was given
        proportion = float(args.examples) / full_dataset_length
        print_verbose(f"Calculated proportion from desired number of examples as {proportion:.{PRECISION}f}.")  # VERBOSE
    else:  # args.proportion specified, set proportion directly
        proportion = args.proportion

    # generate subset
    if not silent and not verbose:
        sys.stdout.write("Generating subset")
        sys.stdout.flush()
    print_verbose("Generating subset.")  # VERBOSE

    subset = pd.DataFrame()
    for file in files:
        print_verbose(f"Generating subset from file {file}")  # VERBOSE
        dataset = pd.read_csv(file, lineterminator='\n')
        # sample the dataset with stratified (representative) sampling
        temp_subset = dataset.groupby("label", group_keys=False).sample(frac=proportion, random_state=kSEED)
        print_verbose(f"""Generated subset from file {file} with {len(temp_subset)} examples, \
{num_positive_examples(temp_subset)} positive and {num_negative_examples(temp_subset)} negative.""")  # VERBOSE
        # add the sampled subset to the subset dataframe
        subset = pd.concat([subset, temp_subset[["body", "label"]]], ignore_index=True)

        # print progress message
        if not silent and not verbose:
            sys.stdout.write(".")
            sys.stdout.flush()
    if not silent and not verbose: sys.stdout.write("\n"); sys.stdout.flush()

    # if examples was specified, ensure subset has the correct amount of examples
    if args.examples and len(subset.index) < args.examples:  # subset has less examples than required
        print_verbose(f"Resulting subset has less examples than required ({len(subset)} vs {args.examples}). Adding examples.")  # VERBOSE
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
        print_verbose(f"Resulting subset has more examples than required ({len(subset)} vs {args.examples}). Removing examples.")  # VERBOSE
        num_excess_examples = len(subset.index) - args.examples
        # Remove num_excess_examples random rows from the subset
        subset = subset.sample(n=args.examples, random_state=kSEED).reset_index(drop=True)

    # print information about subset
    if not silent:
        print(f"""Finished generating subset with {len(subset)} examples \
and a positive proportion of {proportion_positive_examples(subset):.{PRECISION}f} \
from a dataset with {full_dataset_length} examples \
and a positive proportion of {full_dataset_positive_proportion:.{PRECISION}f}.""", flush=True)

    # write subset to csv without row indices
    print_verbose(f"Writing subset to file {args.outfile}.")  # VERBOSE
    subset.to_csv(args.outfile, index=False)

    # verify that the csv has the correct number of examples
    if args.verify:
        print_verbose(f"Verifying that {args.outfile} has the correct number of examples.")  # VERBOSE
        dataset = pd.read_csv(args.outfile)

        # verify number of examples
        if args.examples and len(dataset) != args.examples:
            raise ValueError(f"{args.outfile} has {len(dataset)} examples but should have {args.examples} examples.")
        elif args.proportion and not math.isclose(len(dataset), full_dataset_length * args.proportion, rel_tol=VERIFICATION_CLOSENESS):
            raise ValueError(f"{args.outfile} has {len(dataset)} examples but should have about {full_dataset_length * args.proportion:.{PRECISION}f} examples.")
        else:
            if not silent: print(f"Verified that {args.outfile} has the correct number of examples.")

        # verify proportion of examples
        subset_positive_proportion = proportion_positive_examples(dataset)
        if not math.isclose(subset_positive_proportion, full_dataset_positive_proportion, rel_tol=VERIFICATION_CLOSENESS):
            raise ValueError(f"{args.outfile} has a positive proportion of {subset_positive_proportion} but should have a positive proportion of about {full_dataset_positive_proportion}.")
        else:
            if not silent: print(f"Verified that {args.outfile} has the correct proportion of positive examples to negative examples.")
