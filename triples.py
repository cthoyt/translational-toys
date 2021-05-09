import unittest
from itertools import count, repeat
from typing import Any, Iterable, List, Tuple, cast

from more_itertools import chunked, pairwise


def from_tuples(
    triples: Iterable[Tuple[int, int, int]], create_inverse_triples: bool = False
):
    """Create a triples factory from tuples."""
    import torch
    from pykeen.triples import CoreTriplesFactory

    mapped_triples = cast(
        torch.LongTensor, torch.as_tensor(list(triples), dtype=torch.long)
    )
    return CoreTriplesFactory.create(
        mapped_triples, create_inverse_triples=create_inverse_triples
    )


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


def square_grid_factory(rows: int, columns: int, create_inverse_triples: bool = False):
    """Create a square grid in 2D of the given number of rows and columns.

    If you run ``mesh_factory(2, 5)``, you will get the following knowledge graph:

    .. code-block::

         E_0 -[R_0]-> E_1 -[R_0]-> E_2 -[R_0]-> E_3 -[R_0]-> E_4
          |            |            |            |            |
        [R_1]        [R_1]        [R_1]        [R_1]        [R_1]
          ↓            ↓            ↓            ↓            ↓
         E_5 -[R_0]-> E_6 -[R_0]-> E_7 -[R_0]-> E_8 -[R_0]-> E_2
    """
    triples = iter_square_grid_triples(rows=rows, columns=columns)
    return from_tuples(triples, create_inverse_triples=create_inverse_triples)


def iter_square_grid_triples(rows: int, columns: int) -> Iterable[Tuple[int, int, int]]:
    num_entities = rows * columns

    chunks = list(chunked(range(num_entities), rows))
    for chunk in chunks:
        for head, tail in pairwise(chunk):
            yield head, 0, tail

    for chunk in zip(*chunks):
        for head, tail in pairwise(chunk):
            yield head, 1, tail


def hex_grid_factory(rows: int, columns: int, create_inverse_triples: bool = False):
    """Create a hexagonal grid in 2D of the given number of rows and columns.

    :param rows: The number of hexagon rows (if odd, the final row will be a minor row and if even the final row
        will be an even row
    :param columns: The minor row width (major rows have rows + 1)
    :param create_inverse_triples:
    :return: A triples factory for PyKEEN

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
    triples = iter_hex_grid_triples(rows=rows, columns=columns)
    return from_tuples(triples, create_inverse_triples=create_inverse_triples)


def iter_hex_grid_triples(
    rows: int,
    columns: int,
    labels: Tuple[Any, Any, Any] = (0, 1, 2),
) -> Iterable[Tuple[int, int, int]]:
    """Create a hexagonal grid in 2D.

    :param rows: The number of hexagon rows (if odd, the final row will be a minor row and if even the final row
        will be an even row
    :param columns: The minor row width (major rows have rows + 1)
    :param labels: The labels for the left, right, and vertical relation.
    """
    left, right, vert = labels
    for r1, r2 in pairwise(hex_grid(rows, columns)):
        if len(r1) == len(r2):  # minor/minor or major/major
            yield from zip(r1, repeat(vert), r2)
        elif len(r1) < len(r2):  # minor/major
            yield from zip(r1, repeat(left), r2)
            yield from zip(r1, repeat(right), r2[1:])
        else:  # major/minor
            yield from zip(r1, repeat(right), r2)
            yield from zip(r1[1:], repeat(left), r2)


def hex_grid(rows: int, columns: int) -> List[List[int]]:
    rv = []
    counter = count()

    def _append_row(n: int) -> None:
        rv.append([next(counter) for _ in range(n)])

    # First row is special - always is the first quarter of a minor row
    _append_row(columns)

    for row in range(rows):
        for _ in range(2):  # double up rows for cross beams
            _append_row(columns + 1 + row % 2)

    # Last row is special, is columns + 1 if # rows is even, else columns
    _append_row(columns + (1 + rows) % 2)

    return rv


class TestFactories(unittest.TestCase):
    def test_hex_rows(self):
        for rows, columns, expected in [
            # one row
            (
                1,
                1,
                [[0], [1, 2], [3, 4], [5]],
            ),
            (1, 2, [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9]]),
            (
                1,
                3,
                [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13]],
            ),
            # two rows
            (
                2,
                1,
                [
                    [0],
                    [1, 2],
                    [3, 4],
                    [5, 6, 7],
                    [8, 9, 10],
                    [11, 12],
                ],
            ),
            (
                2,
                2,
                [
                    [0, 1],
                    [2, 3, 4],
                    [5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                    [16, 17, 18],
                ],
            ),
            (
                2,
                3,
                [
                    [0, 1, 2],
                    [3, 4, 5, 6],
                    [7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24],
                ],
            ),
            # three rows
            (
                3,
                1,
                [
                    [0],
                    [1, 2],
                    [3, 4],
                    [5, 6, 7],
                    [8, 9, 10],
                    [11, 12],
                    [13, 14],
                    [15],
                ],
            ),
            (
                3,
                2,
                [
                    [0, 1],
                    [2, 3, 4],
                    [5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                    [16, 17, 18],
                    [19, 20, 21],
                    [22, 23],
                ],
            ),
            (
                3,
                3,
                [
                    [0, 1, 2],
                    [3, 4, 5, 6],
                    [7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31],
                ],
            ),
            # four rows
            (
                4,
                1,
                [
                    [0],
                    [1, 2],
                    [3, 4],
                    [5, 6, 7],
                    [8, 9, 10],
                    [11, 12],
                    [13, 14],
                    [15, 16, 17],
                    [18, 19, 20],
                    [21, 22],
                ],
            ),
        ]:
            with self.subTest(rows=rows, columns=columns):
                grid = hex_grid(rows=rows, columns=columns)
                print(rows, columns, grid)
                self.assertEqual(
                    columns, len(grid[0]), msg="first row have length of columns"
                )
                self.assertEqual(
                    columns + (rows + 1) % 2,
                    len(grid[-1]),
                    msg="# rows even -> columns + 1 in last row; odd -> columns in last row",
                )
                self.assertEqual(
                    2 * (1 + rows),
                    len(expected),
                    msg="test data is wrong: grid should be 2 * (1 + rows) long",
                )
                self.assertEqual(
                    len(expected), len(grid), msg="grid should be 2 * (1 + rows) long"
                )
                self.assertEqual(expected, grid)
