# Semantically-Driven-Sentence-Fusion
Official code for our findings of EMNLP'20 paper "Semantically Driven Sentence Fusion: Modeling and Evaluation" - [[Link]](https://www.researchgate.net/publication/344505509_Semantically_Driven_Sentence_Fusion_Modeling_and_Evaluation). 


If you use this code please cite our paper.

Sentence fusion is the task of joining related sentences into coherent text.  
Current training and evaluation schemes for this task are based on single reference ground-truths and do not account  for  valid  fusion  variants.   
We  show that this hinders models from robustly capturing  the  semantic  relationship  between  input sentences.  
To alleviate this, we present an approach in which ground-truth solutions are automatically expanded into multiple references via curated equivalence classes of connective phrases. We apply this method to a large-scale dataset and use the augmented dataset for both model  training  and  evaluation.    
To  improve the learning of semantic representation using multiple references, we enrich the model with auxiliary  discourse  classification  tasks  undera multi-tasking framework.  



This code is built on Python 3, tensorflow and Google's Pre-trained BERT. Our basic model implementation is highly based on [the implementation in this repository](https://github.com/santhoshkolloju/Abstractive-Summarization-With-Transfer-Learning).  
It works with CPU and GPU.
Our initial data, which we enrich in this work, is taken from [DiscoFuse](https://www.aclweb.org/anthology/N19-1348/).


## Usage Instructions

Running an experiment consists of the following steps (for each of the data-sets, wik & sports):

1. Preprocess data to create an appropriate *.tf_record* files.
2. Train a model on its appropriate data (either from scratch or from a checkpoint).
3. Create test predictions from a traied model.
4. Evaluate test predictions based on single-reference and multi-reference evauluation measures. 

Next we go through these steps on the wiki dataset as a
running example. We use a specific set of hyperparamers, which is defined in the config files. 

### 0.0. Clone/Download this Repository 
Make sure you either clone or download this exact repository, including all directories and code files in your project.

### 0.1. Setup a virtual env 
Make sure you meet all requirements in 'requirements.txt'.

### 0.2. Download Data, pre-trained models and BERT checkpoint.
First, download BERT-base-uncased pretrained weights using the following command: 
```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip 
unzip uncased_L-12_H-768_A-12.zip
```
This will create a directory named *uncased_L-12_H-768_A-12* in your main project directory.

Next, go to [this link](https://mega.nz/folder/QXoilKiJ#p-BX0kZxr4K9hCnwRUAtIg) and download the pre-trained models'-checkpoints: *AuxBert* and *AugAuxBert* (both trained on wiki's train-set).
Place these checkpoints under in the following paths:
1. AuxBert - *models/AuxBert/wiki/*
2. AugAuxBert - *models/AugAuxBert/wiki/*

Now, download all data-sets from [this link](https://mega.nz/folder/xHwy3axK#FzXYfoO0EncDB5ljUJ12ow).
Place the exact same file-structure under *data_frames/*.



### 1. Preprocess data for each model

We are about to train two models: AuxBert and AugAuxBert, which are described in details in our paper.
While AuxBert is trained on the original DiscoFuse train-set (Balanced-wiki in this example), AugAuxBert is trained on an augmented train-set. 
Thus, we need to create different train-set to each.  

Run the following commands

```
# First, create tf_records for AuxBert.
python preprocesses/preprocess_with_auxiliary_tasks.py

# Next, create tf_records for AugAuxBert.
python preprocesses/preprocess_aug_data.py
```

The first command will save tf_records (train, eval, and test) under *data/gen+type+conn/wiki*, and the second command will save same records under  *data/gen+type+conn_large/wiki*.


### 2. Train models - *AuxBert* and *AugAuxBert*.

Since this takes long time, you can choose one of the following:
1. Skip this part and just use our trained models for inference.
2. Use our trained models for initiation. In this case, set the param *total_epochs* in the appropriate config file (*config_with_auxiliary_tasks.py* for *AuxBert* and *config_aug_data.py* for *AugAuxBert*) to be 1 (or any low number).
3. If you don't want to use our trained models, make sure that model directories are empty (fro example, *models/AuxBert/wiki* when training *AuxBert*).

For training *AuxBert*, run the following command:
```
python main_with_auxiliary_tasks.py
```

For training *AugAuxBert*, run the following command:
```
python main_aug_data.py
```

This code saves the ,odel with the highest developement score in 'model/${MODEL_NAME}/wiki/'.


### 3. Create test predictions from a traied model.

In this step we use the trained models to generate fusions to the test set.
This will create the following files in our model's directory (e.g., *models/AuxBert/wiki* when the model is *AuxBert*):
1. tmp.test.src and tmp.test.trg - pairs of gold-fusion and generated-fusions.
2. gt-type-test.pkl and pred-type-test.pkl - discourse phenomena gold-labels and predictions respectively.
3. gt-conns-test.pkl and pred-conns-test.pkl - discourse connectives gold-labels and predictions respectively.

To get *AuxBert* predictions, run the following command:
```
python test/test_with_auxiliary_tasks.py
```

To get *AugAuxBert* predictions, run the following command:
```
python test/test_aug_data.py 
```

### 4. Evaluate test predictions - calculate SARI, MR-SARI, EXACT, and MR-EXACT.

First, to set function params, run the following lines in your command-line:
```
EVAL_FILE_PATH=models/AugAuxBert/wiki/  #switch to 'models/AuxBert/wiki/' if you want to evaluate AuxBert
MR_FILE_PATH=data_frames/wiki/Balanced-multi-ref/test.pickle
```

Now, just run this this command: 
```
python test/evaluate_model_performance.py \
--eval_file_path=${EVAL_FILE_PATH} \ 
--mr_pkl_file_path=${MR_FILE_PATH}
```

Result will be printed to your console (see example below):
```
--- Results ---
     EXACT - 51.98202247191011
     MR-EXACT - 64.5438202247191
     SARI - 85.17741354775626
     MR-SARI - 89.49634330254368
```

### Running with other configuration/hyperparameters
Since the training stage is pretty long, hyper-parameter tuning is pretty hard.
However, note that most configurations are defined in the config files (inside 'configs' dir).
Thus, in order to change any configurations, edit the appropriate config file.



## How to Cite Our Work
```
@article{DBLP:journals/corr/abs-2010-02592,
  author    = {Eyal Ben{-}David and
               Orgad Keller and
               Eric Malmi and
               Idan Szpektor and
               Roi Reichart},
  title     = {Semantically Driven Sentence Fusion: Modeling and Evaluation},
  journal   = {CoRR},
  volume    = {abs/2010.02592},
  year      = {2020}
}
```
