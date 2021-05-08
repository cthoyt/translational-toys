# translational-toys

This repository contains tools for generating toy knowledge graphs representing interesting geometry that correspond to
the philosophies behind knowledge graph embedding models like TransE.
It uses [PyKEEN](https://github.com/pykeen/pykeen) to train the models and animate the evolution of the entity
embeddings.

## Line

A linear dataset embedded with TransE/SoftPlus Loss by running `python cli.py line`:

<picture>
  <source srcset="line/embedding.webp" type="image/webp">
  <source srcset="line/embedding.gif" type="image/png"> 
  <img src="line/embedding.gif" alt="Embedding of a line in 2D">
</picture>

## Square Grid in 2D

A square grid dataset embedded with TransE/NSSA Loss by running `python cli.py mesh`:

<picture>
  <source srcset="square_grid/embedding.webp" type="image/webp">
  <source srcset="square_grid/embedding.gif" type="image/png"> 
  <img src="square_grid/embedding.gif" alt="Embedding of a square grid in 2D">
</picture>

Additional idea: try training in much higher dimensions, then use ISOMAP to reduce
back down to 2D and see how true it is.

## Hexagonal Grid in 2D

TODO

## Circle in 2D

TODO
