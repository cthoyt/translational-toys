# -*- coding: utf-8 -*-

"""Callbacks for making animated embedding plots."""

import os
import pathlib
import time
from typing import List

import click
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pykeen.models import Model
from pykeen.training import TrainingCallback

HERE = pathlib.Path(__file__).parent.resolve()

__all__ = [
    'EntityPlotCallback',
    'LazyEntityPlotCallback',
]


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
