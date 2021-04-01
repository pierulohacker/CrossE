# CrossE

## INTRODUCTION

Original paper: Interaction Embeddings for Prediction and Explanation in Knowledge Graphs. (WSDM'2019)

## DATASETS

There are three benchmark datasets used in this paper, WN18, FB15k and FB15k-237. 

## RUN
(parameters for *FB15k*, *FB15k-237*, *WN18* are taken from the official paper)

for **FB15k**: `python3 CrossE.py --batch 4000 --data datasets/FB15k/ --dim 300 --eval_per 20 --loss_weight 1e-6 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_FB15k/` 

for **FB15k-237**: `python3 CrossE.py --batch 4000 --data datasets/FB15k-237-swapped/ --dim 100 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/FB15k-237-swapped/`

for **WN18**: `python3 CrossE.py --batch 2048 --data datasets/WN18/ --dim 100 --eval_per 20 --loss_weight 1e-4 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/WN18/`

for **DBpedia15k** `python3 CrossE.py --batch 4000 --data datasets/DBpedia15k/ --dim 100 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_DBpedia15k/`

for **DBpediaYAGO** `python3 CrossE.py --batch 2048 --data datasets/DBpediaYAGO/ --dim 300 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_DBpedia15kYAGO/`


python3 CrossE.py --batch 4000 --data datasets/FB15k-237-swapped/ --dim 100 --eval_per 20 --load_model ./save/FB15k-237-swapped/CrossE_DEFAULT_499.ckpt --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/FB15k-237-swapped/
## Restore training
If you want to restore the training phase over a dataset you need to specify the parameter _--load_model_.

E.g.: if you want to restore the data from the experiment made on the reduced version of FB15K dataset 
`--batch 20 --data datasets/FB15k/reduced1.1/ --dim 2 --eval_per 20 --loss_weight 1e-6 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_FB15k_reduced1.1/ --load_model ./save/save_FB15k_reduced1.1/CrossE_DEFAULT_499.ckpt`

for **FB15k**: `python3 CrossE.py --batch 4000 --data datasets/FB15k/ --dim 300 --eval_per 20 --loss_weight 1e-6 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_FB15k/ --load_model ./save/save_FB15k/CrossE_DEFAULT_340.ckpt` 

for **WN18**: `python3 CrossE.py --batch 2048 --data datasets/WN18/ --dim 100 --eval_per 20 --loss_weight 1e-4 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/WN18/ --load_model ./save/WN18/CrossE_DEFAULT_460.ckpt`

for **DBpedia15k** `python3 CrossE.py --batch 4000 --data datasets/DBpedia15k/ --dim 100 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_DBpedia15k/ --load_model ./save/save_DBpedia15k/CrossE_DEFAULT_160.ckpt`


**Pay attention to the checkpoint file to specify:** in the save folder, you'll see files like these
![img.png](docs_support_files/img.png)
but none of them has to be specified to restore the model; instead, you will just specify a name like _CrossE_DEFAULT_499.ckpt_
without any other extension (TensorFlow will manage it).

#EXPLANATION PROCESS
The explanation process is an  extension of the original project, implemented following the algorithm
of the original paper of CrossE. To execute the explanation process it is necessary to follow these steps:
1. Train the model on the desired dataset
2. Load the model from the last iteration in order to save other useful data for the further steps 
   - for **FB15k-237**: `python3 CrossE.py --batch 4000 --data datasets/FB15k-237-swapped/ --dim 100 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/FB15k-237-swapped/ --load_model ./save/FB15k-237-swapped/CrossE_DEFAULT_499.ckpt`

   - for **FB15k**: `python3 CrossE.py --batch 4000 --data datasets/FB15k/ --dim 300 --eval_per 20 --loss_weight 1e-6 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_FB15k/ --load_model ./save/save_FB15k/CrossE_DEFAULT_499.ckpt` 

   - for **WN18**: `python3 CrossE.py --batch 2048 --data datasets/WN18/ --dim 100 --eval_per 20 --loss_weight 1e-4 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/WN18/ --load_model ./save/WN18/CrossE_DEFAULT_499.ckpt`
   
   - for **DBpedia15k** `python3 CrossE.py --batch 4000 --data datasets/DBpedia15k/ --dim 100 --eval_per 20 --loss_weight 1e-5 --lr 0.01 --max_iter 500 --save_per 20 --save_dir ./save/save_DBpedia15k/ --load_model ./save/save_DBpedia15k/CrossE_DEFAULT_499.ckpt`
   
4. Run the batch_similarity.py file
   - for **FB15k-237**: `python3 explanation/batch_similarity.py --data ./save/FB15k-237-swapped/out_data/pickle/`
     
   - for **WN18**: `python3 explanation/batch_similarity.py  --data ./save/WN18/out_data/pickle/`
     
   - for **DBpedia15k** `python3 explanation/batch_similarity.py  --data ./save/save_DBpedia15k/out_data/pickle/`
   
   - for **FB15k**: `python3 explanation/batch_similarity.py  --data ./save/save_FB15k/out_data/pickle/`
   
5. Run the explanation process
   - for **FB15k-237**: `python3 explanation/explanation.py --data ./save/FB15k-237-swapped/out_data/pickle/ --save_dir explanation/results/FB15k-237-swapped/`
   
   - for **DBpedia15k** `python3 explanation/explanation.py --data ./save/save_DBpedia15k/out_data/pickle/ --save_dir explanation/results/save_DBpedia15k/`

## CITE

If the codes help you or the paper inspire your, please cite following paper:

Wen Zhang, Bibek Paudel, Wei Zhang, Abraham Bernstein and Huajun Chen. Interaction Embeddings for Prediction and Explanation in Knowledge Graphs. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining (WSDM2019).

