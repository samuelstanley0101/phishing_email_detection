# Comparing Model Performance on Phishing Email Detection

Datasets obtained from [figshare](https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108).

Download this repository with `git clone https://github.com/samuelstanley0101/phishing_email_detection`

Note that you must have [git-lfs](https://git-lfs.com/) installed to download the dataset.

## logistic.py

`logistic.py` creates a model using logistic regression from a single CSV dataset containing `body` and `label` columns and outputs the results of testing the model to the terminal. It can use TF-IDF or N-Gram vectorization. Here is its usage:

```bash
python3 logistic.py [OPTIONS] file
```

**--tfidf:** Use TF-IDF vectorization.

**--ngram:** Use N-Gram vectorization.

*Note that by default two models will be trained using TF-IDF and N-Gram vectorization, respectively.*

**file:** The file to run logistic regression on. Must be a CSV with `body` and `label` columns.

Python 3.10.3 is recommended. Requirements are listed in `requirements.txt`.

## make-subset.py

`make-subset.py` creates a single subset of any number of CSV datasets containing `body` and `label` columns and writes it to a file. It only loads one dataset at a time, so it should not require much memory. Here is its usage:

```bash
./make-subset.py --proportion [proportion] --outfile [outfile] [OPTIONS] files
```

**--proportion, -P:** The proportion of the original datasets to put in the subset.

**--examples, -E:** The number of examples which should be in the subset.

*Note that specifying number of examples may result in a subset slightly less proportional to the original dataset than specifying the proportion would be.*

**--outfile, -O:** The file to write the subset to. Should be a CSV.

**-r:** Recursively add all files in subdirectories specified.

**--verify:** Verify that the number of examples and proportion of positive to negative examples in the subset is what's expected.

**--verbose:** Print additional messages while running.

**--silent:** Print no messages while running.

**--precision:** The precision to print floating point numbers to the terminal with. Defaults to 4.

**files:** The files or directories containing datasets to make a subset out of. All files from any directories specified will be used, but any subdirectories within the directories specified will be ignored unless `-r` is used. **If no files are specified, all files (excluding subdirectories) from the `source-datasets` directory will be used.**

*Credit to [Geeks for Geeks](https://www.geeksforgeeks.org/python/stratified-sampling-in-pandas/) for help with stratified sampling.*
