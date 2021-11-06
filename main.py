# -*- coding: utf-8 -*-

"""Generate a linear graph and train TransE on it."""

import os
import pathlib
import time
from typing import List, Optional, Tuple, Type, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pygifsicle
import seaborn as sns
from geometric_graphs import hex_grid_factory, line_factory, square_grid_factory
from more_click import verbose_option
from pykeen.losses import Loss, loss_resolver
from pykeen.models import TransE
from pykeen.training import LCWATrainingLoop, TrainingCallback
from pykeen.triples import CoreTriplesFactory
from pykeen.utils import set_random_seed
from torch.optim import Adam

HERE = pathlib.Path(__file__).parent.resolve()
FIGSIZE = 6, 4

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
@click.option("-y", "--learning_rate", type=float, default=0.60, show_default=True)
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

    np.savetxt(
        directory / "triples.tsv",
        triples_factory.mapped_triples.detach().numpy().astype(int),
        fmt="%d",
        delimiter="\t",
    )

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
        entity_constrainer=None,
        relation_constrainer="normalize",
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
            EntityPlotCallback(
                directory=directory,
                animated_extensions=["gif", "webp"],
                frequency=frequency,
                delay=delay,
                skip_post=skip_post,
            ),
            EntityPlotCallback(
                directory=directory,
                animated_extensions=["gif", "webp"],
                frequency=frequency,
                delay=delay,
                skip_post=skip_post,
                apply_lims=True,
                filename="embeddings_static",
            ),
        ],
    )
    model.save_state(directory / "model.pkl")


class EntityPlotCallback(TrainingCallback):
    def __init__(
        self,
        directory: pathlib.Path,
        static_extension: str = "png",
        animated_extensions: Union[str, List[str]] = "gif",
        frequency: int = 5,
        filename: str = "embedding",
        skip_post: bool = False,
        delay: int = 5,
        apply_lims: bool = False,
    ):
        super().__init__()
        self.directory = directory
        self.filename = filename
        self.subdirectory_name = filename
        self.subdirectory = directory / filename
        self.subdirectory.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.skip_post = skip_post
        self.delay = delay
        self.static_extension = static_extension.lstrip(".")
        self.apply_lims = apply_lims
        self.data = []
        if animated_extensions is None:
            self.animated_extensions = ["gif"]
        elif isinstance(animated_extensions, str):
            self.animated_extensions = [animated_extensions.lstrip(".")]
        else:
            self.animated_extensions = [e.lstrip(".") for e in animated_extensions]

    def post_epoch(self, epoch: int, epoch_loss: float, **_kwargs):
        if epoch % self.frequency:
            return  # only make a plot every self.frequency epochs
        self.training_loop.model.eval()
        entity_data = (
            self.training_loop.model.entity_representations[0]()
            .detach()
            .clone()
            .numpy()
        )
        self.data.append(entity_data)

    def post_train(self, losses: List[float], **_kwargs) -> None:
        # Plot losses
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x, y = zip(*enumerate(losses))
        sns.lineplot(x=x, y=y, ax=ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        fig.savefig(self.directory.joinpath("losses").with_suffix(".svg"))
        plt.close(fig)

        # Plot epochs
        data = np.stack(self.data)
        if self.apply_lims:
            low = data.min(axis=(0, 1))
            high = data.max(axis=(0, 1))
            lims = ((low[0] - 0.5, high[0] + 0.5), (low[1] - 0.5, high[1] + 0.5))
        else:
            lims = None
        for i, (entity_data, epoch_loss) in enumerate(zip(data, losses)):
            plot(
                directory=self.subdirectory,
                entity_data=entity_data,
                epoch=i * self.frequency,
                epoch_loss=epoch_loss,
                extension=self.static_extension,
                lims=lims,
            )
        if self.skip_post:
            return

        # Create animation
        for animated_extension in self.animated_extensions:
            click.secho(f"Start making {animated_extension} ({time.asctime()})")
            path = self.directory.joinpath(
                f"{self.filename}.{animated_extension}"
            ).as_posix()
            os.system(
                f"convert -delay {self.delay} -loop 1 {self.subdirectory}/*.{self.static_extension} {path}"
            )
            click.secho(f"Done making {animated_extension} ({time.asctime()})")

            if animated_extension == "gif":
                click.secho(
                    f"Optimizing {animated_extension} with gifsicle ({time.asctime()})"
                )
                pygifsicle.optimize(path)
                click.secho(
                    f"Done optimizing {animated_extension} with gifsicle ({time.asctime()})"
                )


def plot(
    *,
    directory: pathlib.Path,
    entity_data: np.ndarray,
    epoch: int,
    epoch_loss: float,
    extension: str = "png",
    lims: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> None:
    """Save a figure of the embedding plot for the model at the given epoch.

    :param directory: The directory in which to save the chart
    :param entity_data: The embeddings to be plotted
    :param epoch: The epoch from which the embeddings are plotted
    :param epoch_loss: The loss value accumulated over all sub-batches during the epoch
    :param extension: The file extension to use for saving. Defaults to ``png``.
    :param lims: The limits to impose on the plot, if any
    """
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    p1 = sns.scatterplot(
        x=entity_data[:, 0],
        y=entity_data[:, 1],
        ax=ax,
        s=125,
        hue=sns.color_palette("hls", entity_data.shape[0]),
        legend=False,
    )
    for i, (x, y) in enumerate(zip(entity_data[:, 0], entity_data[:, 1])):
        p1.text(
            x,
            y,
            str(i),
            horizontalalignment="center",
            verticalalignment="center",
            size="xx-small",
            color="black",
        )

    ax.set_title(f"Epoch: {epoch:04}; Loss: {epoch_loss:.04}")
    if lims is not None:
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
    fig.tight_layout()
    extension = extension.lstrip(".")  # don't want double dots
    fig.savefig(directory.joinpath(f"{epoch:04}").with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
