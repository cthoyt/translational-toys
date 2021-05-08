import unittest
from itertools import cycle, repeat
from typing import Iterable, Tuple, cast

from more_itertools import chunked, pairwise


def iter_square_grid_triples(rows: int, columns: int) -> Iterable[Tuple[int, int, int]]:
    num_entities = rows * columns

    chunks = list(chunked(range(num_entities), rows))
    for chunk in chunks:
        for head, tail in pairwise(chunk):
            yield head, 0, tail

    for chunk in zip(*chunks):
        for head, tail in pairwise(chunk):
            yield head, 1, tail


def hex_grid(rows: int, columns: int):
    # Make recursive?
    # if 1 < rows:
    #     rv = hex_rows(rows=rows - 1, columns=columns)
    #     # Check the size of the last row
    #     if rows % 2: # odd number of rows
    #         pass
    #     else:  # even number of rows
    #         pass

    rv = []
    index = 0

    # First row is special - always is the first quarter of a minor row
    row = []
    for _ in range(columns):
        row.append(index)
        index += 1
    rv.append(row)

    for _, offset in zip(range(rows), cycle([0, 1])):
        for _ in range(2):  # double up rows for cross beams
            row = []
            for _ in range(columns + offset):
                row.append(index)
                index += 1
            rv.append(row)

    # Last row is special, but is handled differently depending on the number of rows
    if rows % 2:
        row = []
        for _ in range(columns):
            row.append(index)
            index += 1
        rv.append(row)
    else:
        row = []
        for _ in range(columns):
            row.append(index)
            index += 1
        rv.append(row)

    return rv


class TestFactories(unittest.TestCase):
    def test_hex_rows(self):
        for r, c, e in [
            # one row
            (1, 1, [[0], [1, 2], [3, 4], [5]]),
            (1, 2, [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9]]),
            (
                1, 3,
                [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13]],
            ),
            # two rows
            (2, 1, [[0], [1, 2], [3, 4], [5, 6, 7], [8, 9, 10], [11, 12]]),
            (2, 2, [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18]]),
            (
                2, 3,
                [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24]],
            ),
            # three rows
            (3, 1, [[0], [1, 2], [3, 4], [5, 6, 7], [8, 9, 10], [11, 12], [13, 14], [15]]),
            (
                3, 2,
                [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23]],
            ),
            # four rows
        ]:
            with self.subTest(rows=r, columns=c):
                print(r, c, hex_grid(r, c))
                self.assertEqual(e, hex_grid(r, c))


def line_factory(num_entities: int, create_inverse_triples: bool = False):
    """Create a triples factory on a line of ``num_entities`` elements.

    If you run ``line_factory(5)``, you will get the following knowledge graph:

    .. code-block::

        E_0 -[R_0]-> E_1 -[R_0]-> E_2 -[R_0]-> E_3 -[R_0]-> E_4

    If you run ``line_factory(5, create_inverse_triples=True)``, you will get the following knowledge graph:

    .. code-block::

             [R_0]        [R_0]        [R_0]        [R_0]
           /       ⬊     /       ⬊    /       ⬊    /       ⬊
        E_0          E_1          E_2          E_3          E_4
           ⬉       /     ⬉       /    ⬉       /   ⬉       /
             [R_1]        [R_1]        [R_1]        [R_1]
    """
    triples = []
    for head, tail in pairwise(range(num_entities)):
        triples.append((head, 0, tail))
    return from_tuples(triples, create_inverse_triples)


def mesh_factory(rows: int, columns: int, create_inverse_triples: bool = False):
    """Create a square grid in 2D of the given number of rows and columns.

    If you run ``mesh_factory(2, 5)``, you will get the following knowledge graph:

    .. code-block::

         E_0 -[R_0]-> E_1 -[R_0]-> E_2 -[R_0]-> E_3 -[R_0]-> E_4
          |            |            |            |            |
        [R_1]        [R_1]        [R_1]        [R_1]        [R_1]
          ↓            ↓            ↓            ↓            ↓
         E_5 -[R_0]-> E_6 -[R_0]-> E_7 -[R_0]-> E_8 -[R_0]-> E_2
    """
    triples = list(iter_square_grid_triples(rows=rows, columns=columns))
    return from_tuples(triples, create_inverse_triples)


def iter_hex_grid_triples(n_rows: int, n_columns: int):
    """Create a hexagonal grid in 2D.

    :param n_rows: The number of hexagon rows (if odd, the final row will be a minor row and if even the final row
        will be an even row
    :param n_columns: The minor row width (major rows have rows + 1)

    If you run ``hex_factory(rows=1, columns=3)``, you will get the following knowledge graph:

    .. code-block::

                    E_0                     E_1                     E_2
                  ⬋     ⬊                 ⬋     ⬊                 ⬋     ⬊
             [R_0]       [R_1]       [R_0]       [R_1]       [R_0]       [R_1]
            ⬋                 ⬊     ⬋                 ⬊     ⬋                 ⬊
        E_4                     E_5                     E_6                     E_7
         |                       |                       |                       |
       [R_3]                   [R_3]                   [R_3]                   [R_3]
         ↓                       ↓                       ↓                       ↓
        E_8                     E_9                     E_10                    E_11
            ⬊                 ⬋     ⬊                 ⬋     ⬊                 ⬋
             [R_1]       [R_0]       [R_1]       [R_0]       [R_1]       [R_0]
                 ⬊     ⬋                 ⬊     ⬋                 ⬊     ⬋
                   E_12                    E_13                    E_14
    """
    rows = hex_grid(n_rows, n_columns)
    print(rows)

    for r1, r2 in pairwise(rows):
        if len(r1) == len(r2):  # minor/minor or major/major
            yield from zip(r1, repeat(2), r2)
        elif len(r1) < len(r2):  # minor/major
            yield from zip(r1, cycle([0, 1]), r2)
            yield None  # TODO one extra
        else:  # major/minor
            yield from zip(r1, cycle([1, 0]), r2)
            yield None  # TODO one extra


def from_tuples(triples: Iterable[Tuple[int, int, int]], create_inverse_triples: bool = False):
    """Create a triples factory from tuples."""
    import torch
    from pykeen.triples import CoreTriplesFactory
    mapped_triples = cast(torch.LongTensor, torch.as_tensor(triples, dtype=torch.long))
    return CoreTriplesFactory.create(mapped_triples, create_inverse_triples=create_inverse_triples)
