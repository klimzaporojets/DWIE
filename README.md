## DWIE: an entity-centric dataset for multi-task document-level information extraction

### Introduction
DWIE (Deutsche Welle corpus for Information Extraction) is a new dataset for document-level multi-task Information Extraction (IE). 
It combines four main IE sub-tasks: 
1. _Named Entity Recognition_: 23,130 entities classified in 311 multi-label entity types (tags). 
2. _Coreference Resolution_: 43,373 entity mentions clustered in 23,130 entities. 
3. _Relation Extraction_: 21,749 annotated relations between entities classified in 
65 multi-label relation types.  
4. _Entity Linking_: the named entities are linked to Wikipedia (version 20181115). 

DWIE is conceived as an _entity-centric_ dataset that describes interactions 
and properties of conceptual entities on the level of the complete 
document. This contrasts with currently dominant _mention-driven_ 
approaches that start from the detection and classification of 
named entity mentions in individual sentences. Also, the dataset was randomly sampled from a news platform (English online content from Deutsche Welle), 
and the annotation scheme was generated to cover that content. This makes the setting more realistic than in datasets with pre-determined annotation schemes, 
and non-uniform sampling of content to obtain balanced annotations.

We hope DWIE will help in promoting research in multi-task information extraction. 
We will be happy to discuss our work, and are open for collaboration in the future.

### Dataset Download and Preprocessing
Publicly available DWIE annotations are located in the `data/annos` directory. 
In order to get the content of each of the annotated articles, it is necessary
to run the following script:  

```
pip install -r requirements_download_dataset.txt
python src/dwie_download.py
```
This script will retrieve the content of the articles using Deutsche Welle web service, 
add it to the annotation files, and save it in `data/annos_with_content` directory. 
It will also check that the hash representations of the downloaded articles match 
the hash representations of the articles we use in our experiments
(please contact <klim.zaporojets@ugent.be> if the script outputs error messages).      


### Dataset Format
Each of the annotated articles in `data/annos_with_content` is located in a different .json file 
with the following keys: 
- `id`: unique identifier of the article. 
- `content`: textual content of the article downloaded with `src/dwie_download.py` script.
- `tags`: used to differentiate between `train` and `test` sets of documents. 
- `mentions`: a list of entity mentions in the article each with the following keys:
  - `begin`: offset of the first character of the mention (inside `content` field).       
  - `end`: offset of the last character of the mention (inside `content` field). 
  - `text`: the textual representation of the entity mention.  
  - `concept`: the id of the entity that represents the entity mention
  (multiple entity mentions in the article can refer to the same `concept`).  
  - `candidates`: the candidate Wikipedia links. 
  - `scores`: the prior probabilities of the `candidates` entity links calculated
  on Wikipedia corpus. 
  <!--as defined in [Ganea and Hofmann, 2017](https://arxiv.org/pdf/1704.04920.pdf).-->        
- `concepts`: a list of entities that cluster each of the entity `mentions`. 
Each entity is annotated with the following keys: 
  - `concept`: the unique document-level entity id.
  - `text`: the text of the longest mention that belong to the entity. 
  - `keyword`: indicates whether the entity is a keyword. 
  - `count`: the number of entity mentions in the document that 
   belong to the entity.
  - `link`: the entity link to Wikipedia. 
  - `tags`: multi-label classification labels associated to the entity.
- `relations`: a list of document-level relations between entities (`concepts`). 
Each of the relations is annotated with the following keys: 
  - `s`: the subject entity id involved in the relation. 
  - `p`: the predicate that defines the relation name (i.e., "citizen_of", "member_of", etc.).
  - `o`: the object entity id involved in the relation.  
- `iptc`: multi-label article IPTC classification codes. For detailed 
meaning of each of the codes, please refer to the official [IPTC](https://iptc.org/) code list.


### Models
To train and run the models from our paper, we recommend creating a separate environment. 
For example:
```
conda create -n dwie-paper python=3.8
conda activate dwie-paper
pip install -r requirements.txt
``` 
Next, initial files (e.g., embeddings) and configurations have to be set up by running
the following script: 
```
./setup_models.sh
```
Make sure that the directory structure is as follows:
```
├── experiments
├── data
│   ├── annos
│   ├── annos_with_content
│   └── embeddings
├── src
```
 

#### Training
The training script is located in ```src/model_train.py```, it takes two arguments:
 
1- ```--config_file```: the configuration file to run one of the experiments in ```experiments``` directory
 (the names of experiment config files are self explanatory and correspond to Table 9 of the
 paper).  
2- ```--path```: the directory where the output model, results and tensorboard logs are going to be 
saved. For each experiment, a separate subdirectory with the name of experiment is automatically 
created. 

Example: 

```python src/model_train.py --config_file experiments/01_single_coref.json --path results``` 

#### Evaluating
After the _training_ (see above) is finished, the prediction files are saved as ```test.json``` inside the results
directory. In order to obtain the F1 scores execute:  

```python -u src/dwie_evaluation.py --pred [path to prediction file] --gold data/annos/ --gold-filter test```

Example: 

```python -u src/dwie_evaluation.py --pred results/01_single_coref/test.json --gold data/annos/ --gold-filter test```

The ```test.json``` can be also obtained by explicitly loading and evaluating the model in the results directory: 

```python -u src/model_predict.py --path [path to the results directory with serialized model savemodel.pth]```

Example:  

```python -u src/model_predict.py --path results/01_single_coref/```
 

### Evaluation Script
We provide the evaluation script `src/dwie_evaluation.py` in order to obtain all the metrics defined in 
[our paper](https://arxiv.org/abs/2009.12626). The unit test cases are located in `src/tests/` directory. 
The following is an illustrative example of how to use the evaluation script
on one predicted (`predicted.json`) and the respective 
ground truth (`ground_truth.json`) annotation files: 
```
from dwie_evaluation import load_json, EvaluatorDWIE

dwie_eval = EvaluatorDWIE()

loaded_ground_truth = load_json('ground_truth.json', None)
loaded_predicted = load_json('predicted.json', None)

for article_id in loaded_ground_truth.keys():
    dwie_eval.add(loaded_predicted[article_id], loaded_ground_truth[article_id])

# Coreference Metrics
print('Coref MUC F1:', dwie_eval.coref_muc.get_f1())
print('Coref B-Cubed F1:', dwie_eval.coref_bcubed.get_f1())
print('Coref CEAFe F1:', dwie_eval.coref_ceafe.get_f1())
print('Coref Avg.:', sum([dwie_eval.coref_muc.get_f1(), dwie_eval.coref_bcubed.get_f1(), \
                        dwie_eval.coref_ceafe.get_f1()]) / 3)

# NER Metrics
print('NER Mention-Level F1:', dwie_eval.tags_mention.get_f1())
print('NER Hard Entity-Level F1:', dwie_eval.tags_hard.get_f1())
print('NER Soft Entity-Level F1:', dwie_eval.tags_soft.get_f1())

# Relation Extraction (RE) Metrics
print('RE Mention-Level F1:', dwie_eval.rels_mention.get_f1())
print('RE Hard Entity-Level F1:', dwie_eval.rels_hard.get_f1())
print('RE Soft Entity-Level F1:', dwie_eval.rels_soft.get_f1())

```

### Citations
Should you use this code/dataset for your own research, please cite: 
```
@article{ZAPOROJETS2021102563,
title = {{DWIE}: An entity-centric dataset for multi-task document-level information extraction},
journal = {Information Processing & Management},
volume = {58},
number = {4},
pages = {102563},
year = {2021},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2021.102563},
url = {https://www.sciencedirect.com/science/article/pii/S0306457321000662},
author = {Klim Zaporojets and Johannes Deleu and Chris Develder and Thomas Demeester}
}
```

### Contact
If you have questions using DWIE, please e-mail us at <klim.zaporojets@ugent.be>

### Acknowledgements
Part of the research leading to DWIE dataset has received funding from 
(i) the European Union’s Horizon
2020 research and innovation programme under grant agreement no. 761488 for 
the [CPN project](https://www.projectcpn.eu/), and
(ii) the Flemish Government under the "Onderzoeksprogramma Artificiële Intelligentie (AI) Vlaanderen"
programme.

