# -*- coding: utf-8 -*-

"""Generate a linear graph and train TransE on it."""

import pathlib
from typing import Type

import click
from more_click import verbose_option
from torch.optim import Adam

from callbacks import EntityPlotCallback, LazyEntityPlotCallback
from pykeen.losses import Loss, loss_resolver
from pykeen.models import TransE
from pykeen.training import LCWATrainingLoop
from pykeen.triples import CoreTriplesFactory
from pykeen.utils import set_random_seed
from triples import hex_grid_factory, line_factory, square_grid_factory

HERE = pathlib.Path(__file__).parent.resolve()

inverse_option = click.option("--inverse", is_flag=True)


@click.group()
def main():
    """Train a translational model."""


@main.command()
@click.option("-n", "--num-entities", type=int, default=40, show_default=True)
@click.option("-e", "--num-epochs", type=int, default=800, show_default=True)
@loss_resolver.get_option("--loss", default="softplus")
@inverse_option
@verbose_option
def line(num_entities: int, num_epochs: int, loss: Type[Loss], inverse: bool):
    """Train a translational model on a line."""
    triples_factory = line_factory(num_entities, create_inverse_triples=inverse)
    train(
        name="line", triples_factory=triples_factory, num_epochs=num_epochs, loss=loss
    )


@main.command()
@click.option("-r", "--rows", type=int, default=8, show_default=True)
@click.option("-c", "--columns", type=int, default=9, show_default=True)
@click.option("-e", "--num-epochs", type=int, default=600, show_default=True)
@loss_resolver.get_option("--loss", default="nssa")
@inverse_option
@verbose_option
def squares(rows: int, columns: int, num_epochs: int, loss: Type[Loss], inverse: bool):
    """Train a translational model on a square grid."""
    triples_factory = square_grid_factory(
        rows=rows, columns=columns, create_inverse_triples=inverse
    )
    train(
        name="square_grid",
        triples_factory=triples_factory,
        num_epochs=num_epochs,
        loss=loss,
    )


@main.command()
@click.option("-r", "--rows", type=int, default=2, show_default=True)
@click.option("-c", "--columns", type=int, default=5, show_default=True)
@click.option("-e", "--num-epochs", type=int, default=350, show_default=True)
@click.option("-y", "--learning_rate", type=float, default=.60, show_default=True)
@click.option("-d", "--delay", type=float, default=10, show_default=True)
@loss_resolver.get_option("--loss", default="softplus")
@inverse_option
@verbose_option
def hexagons(
    rows: int,
    columns: int,
    num_epochs: int,
    loss: Type[Loss],
    inverse: bool,
    learning_rate: float,
    delay: int,
):
    """Train a translational model on a hexagonal grid."""
    triples_factory = hex_grid_factory(
        rows=rows, columns=columns, create_inverse_triples=inverse
    )
    train(
        name="hexagon_grid",
        triples_factory=triples_factory,
        num_epochs=num_epochs,
        loss=loss,
        learning_rate=learning_rate,
        delay=delay,
    )


def train(
    triples_factory: CoreTriplesFactory,
    num_epochs: int,
    name: str,
    loss: Type[Loss],
    skip_post: bool = False,
    learning_rate: float = 0.5,
    frequency: int = 5,
    delay: int = 5,
) -> None:
    click.secho(f"Training {name}", fg="green", bold=True)
    directory = HERE.joinpath("results", name)
    directory.mkdir(exist_ok=True, parents=True)

    # import numpy as np
    # np.savetxt(
    #     directory / 'triples.tsv',
    #     triples_factory.mapped_triples.detach().numpy().astype(int),
    #     delimiter='\t',
    # )

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
        preferred_device="cpu",
        entity_constrainer=None,  # if you leave this as the default, the entities all just live on the unit circle
        entity_initializer="xavier_uniform",
        # relation_constrainer="normalize",
    )
    optimizer = Adam(
        params=model.get_grad_params(),
        lr=learning_rate,
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
            # EntityPlotCallback(
            #     directory=directory,
            #     animated_extensions=["gif", "webp"],
            #     frequency=frequency,
            #     delay=delay,
            #     skip_post=skip_post,
            # ),
            LazyEntityPlotCallback(
                directory=directory,
                animated_extensions=["gif", "webp"],
                frequency=frequency,
                delay=delay,
                skip_post=skip_post,
            ),
        ],
    )
    model.save_state(directory / "model.pkl")


if __name__ == "__main__":
    main()
