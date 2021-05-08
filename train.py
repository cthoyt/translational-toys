# -*- coding: utf-8 -*-

"""Generate a linear graph and train TransE on it."""

import os
import pathlib
import time
from typing import Iterable, List, Tuple, Type, cast

import click
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from more_click import verbose_option
from more_itertools import chunked, pairwise
from torch.optim import Adam

from pykeen.losses import Loss, loss_resolver
from pykeen.models import Model, TransE
from pykeen.training import LCWATrainingLoop, TrainingCallback
from pykeen.triples import CoreTriplesFactory
from pykeen.utils import set_random_seed

HERE = pathlib.Path(__file__).parent.resolve()


class EntityPlotCallback(TrainingCallback):
    def __init__(self, directory: pathlib.Path, extension: str, frequency: int = 5, subdirectory_name: str = 'img'):
        super().__init__()
        self.directory = directory
        self.subdirectory = directory / subdirectory_name
        self.subdirectory.mkdir(parents=True, exist_ok=True)
        self.extension = extension
        self.frequency = frequency

    def post_epoch(self, epoch: int, epoch_loss: float):
        if epoch % self.frequency:
            return  # only make a plot every self.frequency epochs
        plot(
            directory=self.subdirectory,
            model=self.loop.model,
            epoch=epoch,
            loss=epoch_loss,
            extension=self.extension,
        )

    def post_train(self, losses: List[float]) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x, y = zip(*enumerate(losses))
        sns.lineplot(x=x, y=y, ax=ax)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        fig.savefig(self.directory.joinpath('losses').with_suffix('.svg'))
        plt.close(fig)

        click.secho(f'Start making GIF ({time.asctime()})')
        os.system(f'convert -delay 5 -loop 1 {self.subdirectory}/*.{self.extension} {self.directory}/embedding.gif')
        click.secho(f'Done making GIF ({time.asctime()})')


class LazyEntityPlotCallback(TrainingCallback):
    def __init__(self, directory, frequency: int = 5):
        super().__init__()
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.data = []

    def post_epoch(self, epoch: int, epoch_loss: float):
        if epoch % self.frequency:
            return
        self.loop.model.eval()
        entity_data = self.loop.model.entity_embeddings().detach().numpy()
        self.data.append(entity_data)

    def post_train(self, losses: List[float]):
        data = np.stack(self.data)
        fig, ax = plt.subplots()
        path_collection = plt.scatter([], [])

        def init():
            ax.set_xlim(data[..., 0].min(), data[..., 0].max())
            ax.set_ylim(data[..., 1].min(), data[..., 1].max())
            return path_collection,

        def update(frame: int):
            path_collection.set_offsets(data[frame])
            return path_collection,

        func_animation = matplotlib.animation.FuncAnimation(
            fig,
            update,
            frames=np.arange(len(data)),
            init_func=init,
            blit=True,
        )
        # plt.show()
        func_animation.save(self.directory / "animation.mp4", writer="ffmpeg")
        plt.close(fig)


extension_option = click.option('-x', '--extension', default='png')
inverse_option = click.option('--inverse', is_flag=True)


@click.group()
def main():
    """Train a translational model."""


@main.command()
@click.option('-n', '--num-entities', type=int, default=40, show_default=True)
@click.option('-e', '--num-epochs', type=int, default=2000, show_default=True)
@extension_option
@verbose_option
@loss_resolver.get_option('--loss', default='softplus')
@inverse_option
def line(num_entities: int, num_epochs: int, extension: str, loss: Type[Loss], inverse: bool):
    """Train a translational model on a line."""
    triples_factory = line_factory(num_entities, create_inverse_triples=inverse)
    train(name='line', triples_factory=triples_factory, num_epochs=num_epochs, extension=extension, loss=loss)


@main.command()
@click.option('-r', '--rows', type=int, default=8, show_default=True)
@click.option('-c', '--columns', type=int, default=9, show_default=True)
@click.option('-e', '--num-epochs', type=int, default=600, show_default=True)
@extension_option
@loss_resolver.get_option('--loss', default='nssa')
@inverse_option
@verbose_option
def mesh(rows: int, columns: int, num_epochs: int, extension: str, loss: Type[Loss], inverse: bool):
    """Train a translational model on a mesh."""
    triples_factory = mesh_factory(rows=rows, columns=columns, create_inverse_triples=inverse)
    train(name='mesh', triples_factory=triples_factory, num_epochs=num_epochs, extension=extension, loss=loss)


def train(
    triples_factory: CoreTriplesFactory,
    num_epochs: int,
    name: str,
    extension: str,
    loss: Type[Loss],
):
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
            EntityPlotCallback(directory, extension),
            LazyEntityPlotCallback(directory),
        ],
    )


def plot(*, directory: pathlib.Path, model: Model, epoch: int, loss: float, extension: str):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    model.eval()
    entity_data = model.entity_embeddings().detach().numpy()
    sns.scatterplot(
        x=entity_data[:, 0],
        y=entity_data[:, 1],
        ax=ax,
        hue=sns.color_palette("hls", entity_data.shape[0]),
        legend=False,
    )
    ax.set_title(f'Epoch: {epoch:04}; Loss: {loss:.04}')
    fig.tight_layout()
    fig.savefig(directory.joinpath(f'{epoch:04}').with_suffix(f'.{extension}'), dpi=300)
    plt.close(fig)


def line_factory(num_entities: int, create_inverse_triples: bool = False) -> CoreTriplesFactory:
    """Create a triples factory on a line of ``num_entities`` elements."""
    triples = []
    for head, tail in pairwise(range(num_entities)):
        triples.append((head, 0, tail))
    return _from_triples(triples, create_inverse_triples)


def mesh_factory(rows: int, columns: int, create_inverse_triples: bool = False):
    """Create a mesh of the given number of rows and columns."""
    triples = list(iter_mesh_triples(rows=rows, columns=columns))
    return _from_triples(triples, create_inverse_triples)


def iter_mesh_triples(rows: int, columns: int) -> Iterable[Tuple[int, int, int]]:
    num_entities = rows * columns

    chunks = list(chunked(range(num_entities), rows))
    for chunk in chunks:
        for head, tail in pairwise(chunk):
            yield head, 0, tail

    for chunk in zip(*chunks):
        for head, tail in pairwise(chunk):
            yield head, 1, tail


def _from_triples(triples: Iterable[Tuple[int, int, int]], create_inverse_triples: bool = False):
    mapped_triples = cast(torch.LongTensor, torch.as_tensor(triples, dtype=torch.long))
    return CoreTriplesFactory.create(mapped_triples, create_inverse_triples=create_inverse_triples)


if __name__ == '__main__':
    main()
