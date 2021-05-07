# ATCS
Research project: Mitigating Label Mismatches across Datasets using Meta-Learning (document classification)

## Installation
The environment can be created with conda env create -f atcs2.yml

If there is difficulty with installing the huggingface datasets, do it manually with conda install -c huggingface -c conda-forge datasets

## Code hygiene
Please keep a separate branch for developing each of the three tasks. Once your functionality is completely tested and ready (not v0.3, but v1.0), merge it into the main branch. If you want to add a new functionality to a branch that requires (major) changes to existing code, create a new branch, finish your thing, merge it back.
I recommend github desktop for convenience.
If you want to be really proper: If you want to add **any** new functionality to a branch, create a new branch and follow the same procedure.

For datasets, please maintain/create a structure of 'Data/DatasetName/TrainOrTestOrDevSplit/files'. Everything in the Data folder is ignored, so please add an automatic download function in your code, and/or provide download instructions in this README.

If your functionalities are done, add them to main and make them callable from the command line.

Furthermore, maintain general hygiene:
- Define classes (e,g. lightning models), functionalities (e.g. cleaning the dataset), and experiments (e.g. finetune bert) in separate files.
- Update atcs2.yml file if you install new libraries
- Separate functions as much as possible
- Comment your code!!1!
- stick to the styleguide https://pep8.org/

## Datasets
HuffPost News Category Dataset: This dataset contains 200k documents, each with a headline and a short description. The documents are annotated with a fine grained label of its topic. Due to the size of the dataset, it might be worth to exclude a few labels during training, and use those documents for few-shot evaluation later. 
Link: https://huggingface.co/datasets/Fraser/news-category-dataset 

AG Corpus of News Articles: This dataset contains documents with 4 categories. Check with the related work how you commonly preprocess it. 
Related Work: https://paperswithcode.com/sota/text-classification-on-ag-news, http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html 
Link: https://huggingface.co/datasets/ag_news 


20 Newsgroup: A dataset with 20 categories that are quite different from the previous datasets. Some categories are very closely related while others are rather coarse. This dataset might require some finetuning in terms of which labels to use, and which to merge. 
Link: https://huggingface.co/datasets/newsgroup 


BBC News classification: A small dataset on classification of BBC news articles. 
Link: https://www.kaggle.com/c/learn-ai-bbc 

## BERT
The [BERT-base model from Hugging Face](https://huggingface.co/bert-base-uncased) can finetuned with a single fully connected layer on top on a given dataset. The performance of this model will serve as a baseline for the Meta-Learning and Multitask Learning models.
### Usage
The `train_bert.py` can be called as follows:

``python train_bert.py [-h] [--path PATH] [--optimizer {Adam,SGD}] [--lr LR] [--max_epochs MAX_EPOCHS] [--finetuned_layers FINETUNED_LAYERS] [--tokenizer {BERT}] [--batch-size BATCH_SIZE] [--device {cpu,gpu}] [--seed SEED]
                     [--progress_bar] [--max_text_length MAX_TEXT_LENGTH]
                     name {hp,ag,bbc} nr_classes``

By default the model will use the Adam optimizer to learn only the fully connected layer for 100 epochs with a constant learning rate of `0.001`. Also, by default there is no limit on the text lenght, but the BERT model does not work with text longer than `512` characters.

For more information run `python train_bert.py -h`.
