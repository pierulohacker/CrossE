#!/bin/bash
python3 explanation/explanation.py --data ./save/FB15K/out_data/pickle/ --clustering agglomerative/5 --save_dir explanation/results/WN18/cosine/agglomerative_5/5perc/ --distance cosine --predictions_perc 5
python3 explanation/explanation.py --data ./save/FB15K/out_data/pickle/ --clustering agglomerative/10 --save_dir explanation/results/WN18/cosine/agglomerative_10/5perc/ --distance cosine --predictions_perc 5
python3 explanation/explanation.py --data ./save/FB15K/out_data/pickle/ --clustering agglomerative/15 --save_dir explanation/results/WN18/cosine/agglomerative_15/5perc/ --distance cosine --predictions_perc 5
python3 explanation/explanation.py --data ./save/FB15K/out_data/pickle/ --clustering agglomerative/20 --save_dir explanation/results/WN18/cosine/agglomerative_20/5perc/ --distance cosine --predictions_perc 5
 python3 explanation/explanation.py --data ./save/DBpedia15k/out_data/pickle/ --clustering agglomerative/10 --save_dir explanation/results/DBpedia15k/semantic/agglomerative_10/5perc/ --distance semantic --predictions_perc 5
