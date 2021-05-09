# -*- coding: utf-8 -*-

"""Callbacks for making animated embedding plots."""

import os
import pathlib
import time
from typing import List, Union

import click
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import pygifsicle
import seaborn as sns

from pykeen.models import Model
from pykeen.training import TrainingCallback

__all__ = [
    "EntityPlotCallback",
    "LazyEntityPlotCallback",
]

FIGSIZE = 6, 4


class EntityPlotCallback(TrainingCallback):
    def __init__(
        self,
        directory: pathlib.Path,
        static_extension: str = "png",
        animated_extensions: Union[str, List[str]] = "gif",
        frequency: int = 5,
        subdirectory_name: str = "img",
        filename: str = "embedding",
        skip_post: bool = False,
        delay: int = 5,
    ):
        super().__init__()
        self.directory = directory
        self.subdirectory_name = subdirectory_name
        self.subdirectory = directory / subdirectory_name
        self.subdirectory.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.filename = filename
        self.skip_post = skip_post
        self.delay = delay
        self.static_extension = static_extension.lstrip(".")
        if animated_extensions is None:
            self.animated_extensions = ["gif"]
        elif isinstance(animated_extensions, str):
            self.animated_extensions = [animated_extensions.lstrip(".")]
        else:
            self.animated_extensions = [e.lstrip(".") for e in animated_extensions]

    def post_epoch(self, epoch: int, epoch_loss: float):
        if epoch % self.frequency:
            return  # only make a plot every self.frequency epochs
        plot(
            directory=self.subdirectory,
            model=self.loop.model,
            epoch=epoch,
            epoch_loss=epoch_loss,
            extension=self.static_extension,
        )

    def post_train(self, losses: List[float]) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x, y = zip(*enumerate(losses))
        sns.lineplot(x=x, y=y, ax=ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        fig.savefig(self.directory.joinpath("losses").with_suffix(".svg"))
        plt.close(fig)

        if self.skip_post:
            return
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
        fig, ax = plt.subplots(figsize=FIGSIZE)
        path_collection = plt.scatter([], [])

        def init():
            ax.set_xlim(data[..., 0].min(), data[..., 0].max())
            ax.set_ylim(data[..., 1].min(), data[..., 1].max())
            return (path_collection,)

        def update(frame: int):
            path_collection.set_offsets(data[frame])
            return (path_collection,)

        func_animation = matplotlib.animation.FuncAnimation(
            fig,
            update,
            frames=np.arange(len(data)),
            init_func=init,
            blit=True,
        )
        # plt.show()
        func_animation.save(self.directory / "embeddings.mp4", writer="ffmpeg")
        plt.close(fig)


def plot(
    *,
    directory: pathlib.Path,
    model: Model,
    epoch: int,
    epoch_loss: float,
    extension: str = "png",
) -> None:
    """Save a figure of the embedding plot for the model at the given epoch.

    :param directory: The directory in which to save the chart
    :param model: The model whose embeddings will be plotted
    :param epoch: The epoch from which the embeddings are plotted
    :param epoch_loss: The loss value accumulated over all sub-batches during the epoch
    :param extension: The file extension to use for saving. Defaults to ``png``.
    """
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    model.eval()
    entity_data = model.entity_embeddings().detach().numpy()
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
    fig.tight_layout()
    extension = extension.lstrip(".")  # don't want double dots
    fig.savefig(directory.joinpath(f"{epoch:04}").with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)
