# -*- coding: utf-8 -*-

"""Generate a linear graph and train TransE on it."""

import pathlib
from typing import Type

import click
from more_click import verbose_option
from torch.optim import Adam

from callbacks import EntityPlotCallback
from pykeen.losses import Loss, loss_resolver
from pykeen.models import TransE
from pykeen.training import LCWATrainingLoop
from pykeen.triples import CoreTriplesFactory
from pykeen.utils import set_random_seed
from triples import line_factory, mesh_factory

HERE = pathlib.Path(__file__).parent.resolve()

inverse_option = click.option('--inverse', is_flag=True)


@click.group()
def main():
    """Train a translational model."""


@main.command()
@click.option('-n', '--num-entities', type=int, default=40, show_default=True)
@click.option('-e', '--num-epochs', type=int, default=800, show_default=True)
@loss_resolver.get_option('--loss', default='softplus')
@inverse_option
@verbose_option
def line(num_entities: int, num_epochs: int, loss: Type[Loss], inverse: bool):
    """Train a translational model on a line."""
    triples_factory = line_factory(num_entities, create_inverse_triples=inverse)
    train(name='line', triples_factory=triples_factory, num_epochs=num_epochs, loss=loss)


@main.command()
@click.option('-r', '--rows', type=int, default=8, show_default=True)
@click.option('-c', '--columns', type=int, default=9, show_default=True)
@click.option('-e', '--num-epochs', type=int, default=600, show_default=True)
@loss_resolver.get_option('--loss', default='nssa')
@inverse_option
@verbose_option
def mesh(rows: int, columns: int, num_epochs: int, loss: Type[Loss], inverse: bool):
    """Train a translational model on a mesh."""
    triples_factory = mesh_factory(rows=rows, columns=columns, create_inverse_triples=inverse)
    train(name='square_grid', triples_factory=triples_factory, num_epochs=num_epochs, loss=loss)


def train(
    triples_factory: CoreTriplesFactory,
    num_epochs: int,
    name: str,
    loss: Type[Loss],
) -> None:
    directory = HERE.joinpath(name)

    # Set the random seed for all experiments
    random_seed = 0
    set_random_seed(random_seed)

    # Prepare model and trainer
    model = TransE(
        triples_factory=triples_factory,
        loss=loss(),
        scoring_fct_norm=1,
        embedding_dim=2,
        random_seed=random_seed,
        preferred_device='cpu',
        entity_constrainer=None,  # if you leave this as the default, the entities all just live on the unit circle
        entity_initializer='xavier_uniform',
    )
    optimizer = Adam(
        params=model.get_grad_params(),
        lr=0.5,
    )
    trainer = LCWATrainingLoop(
        triples_factory=triples_factory,
        model=model,
        optimizer=optimizer,
        automatic_memory_optimization=False,  # not necessary on CPU
    )
    trainer.train(
        triples_factory=triples_factory,
        num_epochs=num_epochs,
        batch_size=256,
        callbacks=[
            EntityPlotCallback(directory=directory, animated_extensions=['gif', 'webp']),
            # LazyEntityPlotCallback(directory),
        ],
    )


if __name__ == '__main__':
    main()
