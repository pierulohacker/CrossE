# CrossE

## INTRODUCTION

Original paper: Interaction Embeddings for Prediction and Explanation in Knowledge Graphs. (WSDM'2019)

## DATASETS

There are three benchmark datasets used in this paper, WN18, FB15k and DBPedia15k. 

## RUN
(parameters for *FB15k*, *FB15k-237*, *WN18* are taken from the official paper)

for **FB15k**: `python3 CrossE.py --batch 4000 --data datasets/FB15k/ --dim 300 --eval_per 20 --loss_weight 1e-6 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_FB15k/` 

for **FB15k-237**: `python3 CrossE.py --batch 4000 --data datasets/FB15k-237-swapped/ --dim 100 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/FB15k-237-swapped/`

for **WN18**: `python3 CrossE.py --batch 2048 --data datasets/WN18/ --dim 100 --eval_per 20 --loss_weight 1e-4 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/WN18/`

for **DBpedia15k** `python3 CrossE.py --batch 4000 --data datasets/DBpedia15k/ --dim 100 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_DBpedia15k/`

for **DBpediaYAGO** `python3 CrossE.py --batch 1000 --data datasets/DBpediaYAGO/ --dim 300 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_DBpediaYAGO/`
python3 explanation/explanation.py --data ./save/DBpedia15k/out_data/pickle/ --save_dir explanation/results/DBpedia15k/euclidian/2perc/ --multiprocessing True


python3 CrossE.py --batch 4000 --data datasets/FB15k-237-swapped/ --dim 100 --eval_per 20 --load_model ./save/FB15k-237-swapped/CrossE_DEFAULT_499.ckpt --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/FB15k-237-swapped/
## Restore training
If you want to restore the training phase over a dataset you need to specify the parameter _--load_model_.

E.g.: if you want to restore the data from the experiment made on the reduced version of FB15K dataset 
`--batch 20 --data datasets/FB15k/reduced1.1/ --dim 2 --eval_per 20 --loss_weight 1e-6 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_FB15k_reduced1.1/ --load_model ./save/save_FB15k_reduced1.1/CrossE_DEFAULT_499.ckpt`

for **FB15k**: `python3 CrossE.py --batch 4000 --data datasets/FB15k/ --dim 300 --eval_per 20 --loss_weight 1e-6 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_FB15k/ --load_model ./save/save_FB15k/CrossE_DEFAULT_340.ckpt` 

for **WN18**: `python3 CrossE.py --batch 2048 --data datasets/WN18/ --dim 100 --eval_per 20 --loss_weight 1e-4 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/WN18/ --load_model ./save/WN18/CrossE_DEFAULT_460.ckpt`

for **DBpedia15k** `python3 CrossE.py --batch 4000 --data datasets/DBpedia15k/ --dim 100 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_DBpedia15k/ --load_model ./save/save_DBpedia15k/CrossE_DEFAULT_160.ckpt`

for **DBpediaYAGO** `python3 CrossE.py --batch 1000 --data datasets/DBpediaYAGO/ --dim 300 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_DBpediaYAGO/ --load_model ./save/save_DBpediaYAGO/CrossE_DEFAULT_180.ckpt --worker 10`


**Pay attention to the checkpoint file to specify:** in the save folder, you'll see files like these
![img.png](docs_support_files/img.png)
but none of them has to be specified to restore the model; instead, you will just specify a name like _CrossE_DEFAULT_499.ckpt_
without any other extension (TensorFlow will manage it).

#EXPLANATION PROCESS
The explanation process is an extension of the original project, implemented following the algorithm
of the original paper of CrossE. To execute the explanation process it is necessary to follow these steps:
1. Train the model on the desired dataset
2. Load the model from the last iteration in order to save other useful data for the further steps 
   - for **FB15k-237**: `python3 CrossE.py --batch 4000 --data datasets/FB15k-237-swapped/ --dim 100 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/FB15k-237-swapped/ --load_model ./save/FB15k-237-swapped/CrossE_DEFAULT_499.ckpt`

   - for **FB15k**: `python3 CrossE.py --batch 4000 --data datasets/FB15k/ --dim 300 --eval_per 20 --loss_weight 1e-6 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_FB15k/ --load_model ./save/save_FB15k/CrossE_DEFAULT_499.ckpt` 

   - for **WN18**: `python3 CrossE.py --batch 2048 --data datasets/WN18/ --dim 100 --eval_per 20 --loss_weight 1e-4 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/WN18/ --load_model ./save/WN18/CrossE_DEFAULT_499.ckpt`
   
   - for **DBpedia15k** `python3 CrossE.py --batch 4000 --data datasets/DBpedia15k/ --dim 100 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_DBpedia15k/ --load_model ./save/save_DBpedia15k/CrossE_DEFAULT_499.ckpt`
   
3. Run the batch_similarity.py file
    - EUCLIDIAN DISTANCE (standard)
        - for **FB15k-237**: `python3 explanation/batch_similarity.py --data ./save/FB15k-237-swapped/out_data/pickle/`
        - for **WN18**: `python3 explanation/batch_similarity.py  --data ./save/WN18/out_data/pickle/`
        - for **DBpedia15k** `python3 explanation/batch_similarity.py  --data ./save/save_DBpedia15k/out_data/pickle/`
        - for **FB15k**: `python3 explanation/batch_similarity.py  --data ./save/save_FB15k/out_data/pickle/`
   
    - COSINE SIMILARITY
        - for **FB15k**: `python3 explanation/batch_similarity.py  --data ./save/save_FB15k/out_data/pickle/ --distance cosine`
        - for **FB15k-237**: `python3 explanation/batch_similarity.py --data ./save/FB15k-237-swapped/out_data/pickle/ --distance cosine`
        - for **DBpedia15k** `python3 explanation/batch_similarity.py  --data ./save/save_DBpedia15k/out_data/pickle/ --distance cosine`
        - for **WN18**: `python3 explanation/batch_similarity.py  --data ./save/WN18/out_data/pickle/ --distance cosine`
    
    - SEMANTIC SIMILARITY
        - for **DBpedia15k** `python3 explanation/batch_similarity.py --data ./save/save_DBpedia15k/out_data/pickle/ --semantic_data ./datasets/DBpedia15k/`
    
- #### CLUSTERING
    In order to speed up the explanation process, the entity embeddings can be clustered in groups. Given an embedding, the search of the closest vectors
    will be done inside the cluster that embedding belongs to. Without clustering, this search would be conducted in the whole space, requiring more time. <br>
    You can choose between
    **KMeans clustering** and **agglomerative clustering**. For both the techniques, tests were made considering **8**, **10**, **15** clusters, but you can create as many as you want. <br>
  - Create the clustering model by running the file explanation/clustering.py. Suppose we want to group the DBPedia15k entity embeddings in 10 clusters using the kmeans method 
              and want to save them in the `clustering_DBPedia` folder. The command to run is: 
    - `python explanation/clustering.py --path_to_embs save/DBpedia15k/out_data/pickle/ent_emb.pkl --k 10 --dest clustering_DBPedia/ --type kmeans`
  - Now suppose that we want to use the previously created clustering model and save the results in `save/DBpedia15k/out_data/pickle/kmeans_8/euclidian/`. The command to run is
    - `python3 explanation/batch_similarity.py --data ./save/DBpedia15k/out_data/pickle/ --save_dir ./save/DBpedia15k/out_data/pickle/kmeans_8/euclidian/ --clustering clustering_DBPedia/KMeans_10.pkl`

   
5. Run the explanation process
   
    - **EUCLIDEAN DISTANCE BASED** (default):
        - 2% of predictions (is default):
            - for **FB15k**: `python3 explanation/explanation.py --data ./save/save_FB15k/out_data/pickle/ --save_dir explanation/results/save_FB15k/Euclidian/2perc/`
          
            - for **FB15k-237**: `python3 explanation/explanation.py --data ./save/FB15k-237-swapped/out_data/pickle/ --save_dir explanation/results/FB15k-237-swapped/Euclidian/2perc/`
       
            - for **DBpedia15k** `python3 explanation/explanation.py --data ./save/save_DBpedia15k/out_data/pickle/ --save_dir explanation/results/save_DBpedia15k/Euclidian/2perc/`
          
            - for **WN18** `python3 explanation/explanation.py --data ./save/WN18/out_data/pickle/ --save_dir explanation/results/WN18/Euclidian/2perc/`
       
            - for **DBpediaYAGO** `python3 explanation/explanation.py --data ./save/save_DBpediaYAGO/out_data/pickle/ --save_dir explanation/results/save_DBpediaYAGO/Euclidian/2perc/`
    
        - 5% of predictions
            - for **FB15k**: `python3 explanation/explanation.py --data ./save/save_FB15k/out_data/pickle/ --save_dir explanation/results/save_FB15k/Euclidian/5perc/ --predictions_perc 5`
          
            - for **FB15k-237**: `python3 explanation/explanation.py --data ./save/FB15k-237-swapped/out_data/pickle/ --save_dir explanation/results/FB15k-237-swapped/Euclidian/5perc/ --predictions_perc 5`
       
            - for **DBpedia15k** `python3 explanation/explanation.py --data ./save/save_DBpedia15k/out_data/pickle/ --save_dir explanation/results/save_DBpedia15k/Euclidian/5perc/ --predictions_perc 5`
          
            - for **WN18** `python3 explanation/explanation.py --data ./save/WN18/out_data/pickle/ --save_dir explanation/results/WN18/Euclidian/5perc/ --predictions_perc 5`
       
            - for **DBpediaYAGO** `python3 explanation/explanation.py --data ./save/save_DBpediaYAGO/out_data/pickle/ --save_dir explanation/results/save_DBpediaYAGO/Euclidian/5perc/ --predictions_perc 5`

    - **COSINE SIMILARITY BASED**
        - 2% of predictions (is default):
            - for **FB15k**: `python3 explanation/explanation.py --data ./save/save_FB15k/out_data/pickle/ --save_dir explanation/results/save_FB15k/cosine/2perc/ --distance cosine `
          
            - for **FB15k-237**: `python3 explanation/explanation.py --data ./save/FB15k-237-swapped/out_data/pickle/ --save_dir explanation/results/FB15k-237-swapped/cosine/2perc/ --distance cosine`
       
            - for **DBpedia15k** `python3 explanation/explanation.py --data ./save/save_DBpedia15k/out_data/pickle/ --save_dir explanation/results/save_DBpedia15k/cosine/2perc/ --distance cosine`
          
            - for **WN18** `python3 explanation/explanation.py --data ./save/WN18/out_data/pickle/ --save_dir explanation/results/WN18/cosine/2perc/ --distance cosine`
       
            - for **DBpediaYAGO** `python3 explanation/explanation.py --data ./save/save_DBpediaYAGO/out_data/pickle/ --save_dir explanation/results/save_DBpediaYAGO/cosine/2perc/ --distance cosine`
    
        - 5% of predictions
            - for **FB15k**: `python3 explanation/explanation.py --data ./save/save_FB15k/out_data/pickle/ --save_dir explanation/results/save_FB15k/cosine/5perc/ --predictions_perc 5 --distance cosine`
          
            - for **FB15k-237**: `python3 explanation/explanation.py --data ./save/FB15k-237-swapped/out_data/pickle/ --save_dir explanation/results/FB15k-237-swapped/cosine/5perc/ --predictions_perc 5 --distance cosine`
       
            - for **DBpedia15k** `python3 explanation/explanation.py --data ./save/save_DBpedia15k/out_data/pickle/ --save_dir explanation/results/save_DBpedia15k/cosine/5perc/ --predictions_perc 5 --distance cosine`
          
            - for **WN18** `python3 explanation/explanation.py --data ./save/WN18/out_data/pickle/ --save_dir explanation/results/WN18/cosine/5perc/ --predictions_perc 5 --distance cosine`
       
            - for **DBpediaYAGO** `python3 explanation/explanation.py --data ./save/save_DBpediaYAGO/out_data/pickle/ --save_dir explanation/results/save_DBpediaYAGO/cosine/5perc/ --predictions_perc 5 --distance cosine`
Provare KDtree
Adattare misura di similarità adottata nel clustering alla misura che sarà usata per trovare gli embedding vicini
Tenere traccia del numero di confronti
Fare la ricerca prima in un unico cluster, poi in due
