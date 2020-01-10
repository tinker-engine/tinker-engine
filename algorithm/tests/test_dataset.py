import pytest
from ..dataset import *


@pytest.mark.parametrize(
    "test_image, number_of_bands",
    [
        ('rgb.jpg', 3),
        ('grayscale.png', 3),
        # ('multiband.tif', 7) # Should we fix this issue in case people use tiffs with >3 channels?
    ]
)
def test_pil_loader(shared_datadir, test_image, number_of_bands):
    converted_image = pil_loader(shared_datadir / test_image)
    assert len(converted_image.getbands()) == number_of_bands


@pytest.mark.parametrize(
    "test_image",
    [
        ('rgb.jpg'),
        ('grayscale.png'),
        ('multiband.tif')
    ]
)
def test_is_image_file(test_image):
    is_image_file(test_image)


# @pytest.mark.parametrize(
#     "problem",
#     [
#         "big_problem" # Should we add better error messaging for wrong problem descriptions?
#     ]
# )
# def test_jpl_dataset_initialization(problem):
#     JPLDataset(problem)
