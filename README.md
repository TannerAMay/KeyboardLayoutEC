# Keyboard Layout EC

Generate the "perfect" keyboard layout using evolutionary computing!

Implementation of [this video](https://www.youtube.com/watch?v=EOaPb9wrgDY) by adumb.
Uses the EC Program structure created by my professor and advisor Dr. Taylor for his EC class at Missouri S&T.

## Usage
Use pipenv to create the environment with the Pipfile. If you don't to, the only package you need is tqdm.  
Download the archive.org dataset from [here](https://www.kaggle.com/datasets/Cornell-University/arxiv) and extract to `data/arxiv-metadata-oai-snapshot.json`

If you are using pipenv:
```sh
pipenv shell
pipenv run python kb_ec.py
```

If you are just using normal python:
```sh
python kb_ec.py
```

Only tested on linux.


## Representation
As mentioned in adumb's video the keyboard representation is top left to bottom right, column wise. So QWERTY would be:
`qazwsxedcrfvtgbyhnujmik,ol.p;/`

## Fitness function
The fitness function measures the distance between consecutive key presses. Adumb's video explains it really well.  
Unfortunately they did not include in their screen capture, so I had to improvise and it is quite ugly.

## Recombination
Starting from a random location, a random amount of keys will be copied from the first parent to child. The second parent will fill in the rest
of the keys, without duplicates, according to the order the occur in the second parent's genome.

## Mutation
10% of individuals will have two random keys swapped.
