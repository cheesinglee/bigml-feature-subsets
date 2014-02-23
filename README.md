# Feature Subset Selection using the BigML API

    usage: feature_subsets.py [-h] [-u USERNAME] [-a APIKEY] [-n NFOLDS]
                              [-k STALENESS] [-p PENALTY]
                              filename

    positional arguments:
      filename              path to CSV file

    optional arguments:
      -h, --help            show this help message and exit
      -u USERNAME, --username USERNAME
                            BigML username
      -a APIKEY, --apikey APIKEY
                            BigML API key
      -n NFOLDS, --nfolds NFOLDS
                            Number of cross-validation folds [default=5]
      -k STALENESS, --staleness STALENESS
                            Staleness parameter for best-first search [default=5]
      -p PENALTY, --penalty PENALTY
                            Per-feature penalty factor [default=0.001]

## Dependencies

- Python 2.x
- BigML Python bindings
- scikit-learn
