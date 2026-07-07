import pytest

from segmentation.block_utils import prepare_overlaps


@pytest.mark.parametrize(
    ("diameter", "expected"),
    [
        (None, [12, 12, 12]),
        (15, [15, 15, 15]),
        ((25,), [25, 12, 12]),
        ((25, 25), [25, 25, 12]),
        ((25, 25, 15), [25, 25, 15]),
        ((25, 15, 25, 30), [25, 15, 25]),
    ],
)
def test_prepare_overlaps_using_diameter(diameter, expected):
    assert (
        prepare_overlaps(
            (256, 256, 256),
            (128, 128, 128),
            None,
            default_overlap=diameter,
        )
        == expected
    )
