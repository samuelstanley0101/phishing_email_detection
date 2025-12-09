#!/usr/bin/env bash

mkdir subsets

echo "Generating 1.0 Subset"
./make-subset.py --proportion 1.0 --outfile "subsets/subset_1.csv" source-datasets

echo "Generating 0.1 Subset"
./make-subset.py --proportion 0.1 --outfile "subsets/subset_01.csv" source-datasets

echo "Generating 0.01 Subset"
./make-subset.py --proportion 0.01 --outfile "subsets/subset_001.csv" source-datasets

echo "Generating 0.001 Subset"
./make-subset.py --proportion 0.001 --outfile "subsets/subset_0001.csv" source-datasets