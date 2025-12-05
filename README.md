# Comparing Model Performance on Phishing Email Detection

Datasets obtained from https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108

Download this repository with `git clone https://github.com/samuelstanley0101/phishing_email_detection`

Note that you must have [git-lfs](https://git-lfs.com/) installed to download the dataset.

## make-subset.py

`make-subset.py` creates a single subset of any number of CSV datasets containing `body` and `label` columns and writes it to a file. It only loads one dataset at a time, so it should not require much memory. Here is its usage:

```bash
./make-subset.py --percent [percent] --outfile [outfile] [OPTIONS] files
```

**--percent, -P:** The percent of the original datasets to put in the subset.

**--outfile, -O:** The file to write the subset to. Should be a CSV.

**-r:** Recursively add all files in subdirectories specified.

**files:** The files or directories containing datasets to make a subset out of. All files from any directories specified will be used, but any subdirectories within the directories specified will be ignored unless `-r` is used. **If no files are specified, all files (excluding subdirectories) from the `source-datasets` directory will be used.**

*Note: a --example argument will be added soon.*
