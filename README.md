# NLP-Negative-Sampling - Skip-gram (Word2Vec)
[![CircleCI](https://circleci.com/gh/Driss31/nlp-negative-sampling.svg?style=svg)](https://circleci.com/gh/Driss31/nlp-negative-sampling)
[![codecov](https://codecov.io/gh/Driss31/nlp-negative-sampling/branch/master/graph/badge.svg)](https://codecov.io/gh/Driss31/nlp-negative-sampling)
[![Python: 3.8.2](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://www.python.org/downloads/release/python-381/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

I implement a [Skip Gram model with negative-sampling](https://arxiv.org/pdf/1402.3722.pdf) from scratch. I used the log-likelihood for the loss that I optimized using Mini-Batch Gradient Ascent.

I trained the model on a corpus of words - text file. I generated embedding of useful words then I computed similarity between some words. I also looked for the K most similar words to some pre-defined words that are in the corpus. See below.


## Dependencies

Dependencies are managed by [Poetry](https://python-poetry.org/).

#### Install dependencies

- With dev dependencies:
```bash
$ poetry install
```

- Without dev dependencies:
```bash
$ poetry install --no-dev
```

#### Update dependencies
```bash
$ poetry update
```

## Sanity checks

#### Run the code formatter

This runs [isort](https://github.com/timothycrosley/isort/) and [Black](https://github.com/ambv/black/), the Python code formatter:
```bash
$ make format
```

Black reformats entire files in place, but doesn't reformat blocks that start with `# fmt: off` and end with `# fmt: on`.

#### Run the linter

This test checks syntax error and pip8 rules using [flake8](https://gitlab.com/pycqa/flake8):
```bash
$ make lint
```

#### Run the optional static type checker

This runs `mypy`, the static type checker using the [PEP 484](https://www.python.org/dev/peps/pep-0484/) notation:
```bash
$ make mypy
```

## Model

### Train

Two different methods were tested here. The second had better results.

#### Method 1
Generate one and only one embedding for each word:
For a 1000 unique words corpus, I computed a matrix of 1000 columns where each column contained the embedding of a unique word.

#### Method 2
Generate an embedding for each word once as a target word and again as a context word:
For a 1000 unique words corpus, I generated a matrix of 1000 + 1000 words where I computed an for each unique word, an embedding as a word and an embedding as a context.

#### Command line - Example

```
python skipGram.py --text_path train/news.en-00001-of-00100.txt --model_path train
```
`news.en-00001-of-00100.txt` is used as a training set and the trained model is saved in `train/`.

1. news.en-00001-of-00100.txt  file containing:
- 50 000 sentences.
- 1 094 214 words.
- 5 576 574 positive pairs.
- 27 882 870 negative pairs.
- 14 033 unique words.

```
TRAINING: #epochs: 5, learning_rate: 0.01, batch size: 500, negativeRate: 5, winSize: 7 (+- 3)
Epoch 1/5
100%|████████████████████████████████████| 11153/11153 [39:43<00:00,  4.65it/s]
Epoch 2/5
100%|████████████████████████████████████| 11153/11153 [39:49<00:00,  4.90it/s]
Epoch 3/5
100%|████████████████████████████████████| 11153/11153 [39:59<00:00,  4.50it/s]
Epoch 4/5
100%|████████████████████████████████████| 11153/11153 [39:51<00:00,  4.68it/s]
Epoch 5/5
100%|████████████████████████████████████| 11153/11153 [40:44<00:00,  4.66it/s]
```

If the script crashes, the value of the learning rate should be decreased.

### Evaluation

To compute the similarity between two words, I used the cosine distance.

#### Command line - Example
```
python skipGram.py --text train/news.en-00001-of-00100.txt --model train --test
```

#### Results
```
"woman", "girl"    : 0.9140292408030202
"woman", "man"     : 0.9003719041046656
"woman", "fish"    : 0.6480981482215702
"woman", "bicycle" : 0.5525813805008297
"woman", "green"   : 0.41096331164361916
"woman", "red"     : 0.4022941792833647
"woman", "grizzly" : (Out of Vocabulary -> Grizzly) 0.2520814392860999
```

I computed the similarity between the word `president` and all the words of the corpus, and printed its K=10 most similar words:
```
Similar words to president :
- prime : 0.8675725528021508
- minister : 0.8577274635745533
- barack : 0.8476139485196841
- vice : 0.8338346066461706
- senator : 0.8303469441871172
- presidentelect : 0.8234193469697527
- bush : 0.8217018577730855
- governor : 0.8114519753665658
- john : 0.8096632753090216
- hillary : 0.7903915606521152
```

K=10 most similar words to `financial`:
```
Similar words to financial :
- economic : 0.8605254987941997
- banking : 0.8543459085572801
- housing : 0.8391836329291854
- global : 0.8367757803069268
- institutions : 0.8249920995141922
- development : 0.7974591339002757
- crisis : 0.7927191095780013
- energy : 0.7923018373003
- markets : 0.7898427369024803
- european : 0.7891532793944722
```

10 most similar words to `city`:
```
Similar words to city :
- london : 0.8362202448038254
- virginia : 0.8227038534431752
- west : 0.8210187852324354
- town : 0.8084971923032622
- california : 0.8023458654379504
- iowa : 0.8014968294330261
- angeles : 0.7993281566786211
- park : 0.7968483879597993
- lake : 0.7931283454758526
- southwest : 0.7924832761941711
```


10 most similar words to `barack`:

```
Similar words to barack :
- clinton : 0.8967130891694537
- obama : 0.873172647247945
- hillary : 0.8581969823452125
- president : 0.8476139485196841
- mccain : 0.8474739064219333
- presidentelect : 0.8384489353566097
- bush : 0.8324913896640769
- nominee : 0.8144799174125398
- candidate : 0.8100265607338628
- senator : 0.8096392472156546
```
