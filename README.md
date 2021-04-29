# ACTS
Research project: Mitigating Label Mismatches across Datasets using Meta-Learning (document classification)

## installation
The environment can be created with conda env create -f acts2.yml

If there is difficulty with installing the huggingface datasets, do it manually with conda install -c huggingface -c conda-forge datasets

##Code hygiene
Please keep a separate branch for developing each of the three tasks. Once your functionality is completely tested and ready (not v0.3, but v1.0), merge it into the main branch. If you want to add a new functionality to a branch that requires (major) changes to existing code, create a new branch, finish your thing, merge it back.
I recommend github desktop for convenience.
If you want to be really proper: If you want to add **any** new functionality to a branch, create a new branch and follow the same procedure.

For datasets, please maintain/create a structure of 
'Data/DatasetName/TrainOrTestOrDevSplit/files'

If your functionalities are done, add them to main and make them callable from the command line.

Furthermore, maintain general hygiene:
- Define classes (e,g. lightning models), functionalities (e.g. cleaning the dataset), and experiments (e.g. finetune bert) in separate files.
- Update acts2.yml file if you install new libraries
- Separate functions as much as possible
- Comment your code!!1!
- stick to the styleguide https://pep8.org/
