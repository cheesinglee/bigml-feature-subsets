# Feature Subset Selection using the BigML API

  usage: feature_subsets.py [-h] [-u USERNAME] [-a APIKEY] [-o OBJECTIVE_FIELD]
			    [-t TAG] [-n NFOLDS] [-k STALENESS] [-p PENALTY]
			    [-s SEQUENTIAL]
			    filename

  positional arguments:
    filename              path to CSV file

  optional arguments:
    -h, --help            show this help message and exit
    -u USERNAME, --username USERNAME
			  BigML username
    -a APIKEY, --apikey APIKEY
			  BigML API key
    -o OBJECTIVE_FIELD, --objective_field OBJECTIVE_FIELD
			  Index of objective field [default=last]
    -t TAG, --tag TAG     Tag for created BigML resources [default="Feature selection"]
    -n NFOLDS, --nfolds NFOLDS
			  Number of cross-validation folds [default=5]
    -k STALENESS, --staleness STALENESS
			  Staleness parameter for best-first search [default=5]
    -p PENALTY, --penalty PENALTY
			  Per-feature penalty factor [default=0.001]
    -s SEQUENTIAL, --sequential SEQUENTIAL
			  Perform model building sequentially [default=False]
                            
Example:

    feature_subsets.py --username="my_bigml_username" --apikey="my_bigml_key" data/crx.csv

## Dependencies

- Python 2.x
- BigML Python bindings
- scikit-learn
