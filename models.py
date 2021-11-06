from pykeen.models import ERModel
from pykeen.nn import EmbeddingSpecification
from pykeen.nn.modules import TransEInteraction

__all__ = [
    "TransE",
]


class TransE(ERModel):
    """A modified version of TransE that includes relation normalization"""

    def __init__(self, embedding_dim: int, scoring_fct_norm: int, **kwargs):
        super().__init__(
            interaction=TransEInteraction(p=scoring_fct_norm, power_norm=False),
            entity_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer="xavier_uniform",
                constrainer=None,
            ),
            relation_representations=EmbeddingSpecification(
                embedding_dim=embedding_dim,
                initializer="xavier_uniform_norm",
                constrainer="normalize",
            ),
            **kwargs,
        )
