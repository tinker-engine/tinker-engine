"""
Datasets.

.. _dataset.py:

Kitware manages this function.  The dataset classes for the train/eval parts of
both the train and adapt stages. This will need to change when more types of
problems are added.  Email or create an issue if you want to change
something in this file.
"""

from __future__ import print_function
import torchvision  # type: ignore
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
from PIL import Image  # type: ignore
import warnings
import ubelt as ub  # type: ignore
import torch
import torch.utils.data

from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def basic_transformer() -> Any:
    """
    Resize the image to 32x32 and convert it to a tensor.

    Returns:
        torch.Tensor: image in CHW
    """
    return torchvision.transforms.Compose([torchvision.transforms.Resize([32, 32]), torchvision.transforms.ToTensor()])


def pil_loader(path: str) -> Any:
    """
    Open an image using PIL.

    Args:
        path (str): path to the image

    Return:
        PIL image
    """
    # open path as file to avoid ResourceWarning
    #   (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def has_file_allowed_extension(filename: str, extensions: Tuple[str, str, str, str, str, str, str, str, str]=IMG_EXTENSIONS) -> bool:
    """
    Check if a file is an allowed extension.

    Args:
        filename: path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Check if a file is an allowed image extension.

    Args:
        filename (str): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def ensure_image_list(filelist: List[str]) -> Iterator[str]:
    """
    Take a list of filenames and ensures that these have image extensions.

    Args:
        filelist (list[str]): list of strings containing the file names

    Returns:
        list[str]: containing only the valid image filenames
    """
    return filter(has_file_allowed_extension, filelist)


class ImageClassificationDataset(torchvision.datasets.VisionDataset):
    """
    Training dataset class.

    Contains both labeled and unlabeled images.  Loads images from file. This
    should only be edited by Kitware but feel free to add an issue if you want
    to change something.

    The images are tracked by the indices within this dataset.  For example, if you
    want to request something to be labeled you need to give the
    :meth:`JPLDataset.get_more_labels` method the indices of those images.
    See the method for more info.

    TODO: Update all attributes

    Attributes:
        problem (LwLL): problem class instance containing the
            information,
        name (str): Name of dataset
        image_fnames (list): list of image filenames
        num_images (int): number of images in filelist (both labeled and unlabeled)
        labeled_size (int): number of labeled images in filelist
        unlabeled_size (int): number of unlabeled images in filelist
        labeled_indices (set): set of the indices for the labeled images
        unlabeled_indices (set): set of the indices for the unlabeled images
        targets (list): labels for each image (corresponds with the order for
            image_fnames)
        image_fname_to_index (dict): lookup dict to go from filename to index
        index_to_image_fname (dict): lookup dict to go from index to filename
        categories (list): the list of categories as strings
        category_to_category_index (dict): dict to go from category name to
            category index
        category_index_to_category (dict): dict to go from category index to
            category name

    """

    categories: List[str]
    category_to_category_index: Dict[str, int]
    category_index_to_category: Dict[int, str]
    targets: List[Optional[List[Dict[str, Any]]]]

    def __init__(
        self,
        problem: Any,
        dataset_name: str,
        dataset_root: str,
        transform: Any=ub.NoParam,
        target_transform: Optional[Callable]=None,
        categories: Optional[List[str]]=None,
        seed_labels: Optional[List[str]]=None,
    ) -> None:
        """
        Initialize the dataset.

        This initializes the attributes and gets the seed labels.

        Args:
            problem (LwLL): problem class instance containing the
                information check out problem.py for more information

        Keyword Args:
            baseDataset (bool, optional, default=True): if this is the base
                dataset (as opposed to adapt dataset)
            transform (callable, optional, default=basic_transformer()):
                a function/transform that takes in a sample and returns a
                transformed version.

                E.g. :class:`torchvision.transforms.RandomCrop` for images.

                If no parameter is given, this defaults to
                :obj:`dataset.basic_transformer`
            target_transform (callable, optional, default=None): a function/transform
                that takes in the target and transforms it.


        Raises:
            AssertionError: if dataset size from problem doesn't match what is on
                disk

        """
        self.problem = problem
        self.name = dataset_name
        self.root = dataset_root

        if transform is ub.NoParam:
            transform = basic_transformer()

        super(ImageClassificationDataset, self).__init__(
            self.root, transform=transform, target_transform=target_transform
        )

        self.image_fnames = sorted(ensure_image_list(os.listdir(self.root)))
        # Total number of images in the dataset
        self.num_images = len(self.image_fnames)
        self.labeled_size = 0  # Total number of Labeled Images
        self.unlabeled_size = self.num_images  # Total number of Unlabeled Images

        # targets/label category indices for data
        self.targets = [None] * self.num_images
        self.labeled_indices: Set[int] = set()
        self.unlabeled_indices: Set[int] = set(np.arange(self.num_images).tolist())
        self.index_to_image_fname = dict(zip(np.arange(self.num_images).tolist(), self.image_fnames))
        self.image_fname_to_index = dict(zip(self.image_fnames, np.arange(self.num_images).tolist()))

        # check to make sure number of images is consistent with problem metadata
        self.indices = set(np.arange(self.num_images))
        self.categories = []
        self.category_to_category_index = {}
        self.category_index_to_category = {}

        # TODO: fix the following to correctly initialize
        if categories is None:
            self.get_seed_labels(seed_labels, 0)
        else:
            self.initialize_categories(categories)

    def __len__(self) -> int:
        """
        Get length of unlabeled and labeled data.

        Returns:
            int: number of unlabeled and labeled data
        """
        return self.num_images

    def __getitem__(self, index: int) -> Tuple[Any, int, int]:
        """
        Return an item by index.

        Args:
            index (int): Index

        Returns:
            tuple(PIL Image, int, int): (image, target, index)
                Image in CHW, target is the category index of the target class, and
                index is the index of the image in the dataset
        """
        if isinstance(index, (np.floating, float)):
            index = np.int64(index)

        img_fname, target = self.image_fnames[index], self.targets[index]

        img = pil_loader(os.path.join(self.root, img_fname))
        img = self.transform(img)

        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)

        if target is None:
            # if nothing, make it -1 for the pytorch collate function
            target = -1

        return img, target, index

    def get_unlabeled_indices(self) -> List[int]:
        """
        Return unlabeled indices.

        Returns:
            list (int): indices
                list of indices for unlabeled images
        """
        return list(self.unlabeled_indices)

    def get_labeled_indices(self) -> List[int]:
        """
        Return labeled indices.

        Returns:
            list (int): indices
                list of indices for labeled images
        """
        return list(self.labeled_indices)

    def get_more_labels(self, indices: List[int]) -> None:
        """
        Ask LwLL class to interface with JPL server to query for indices.

        This function will check to make sure all requested images are unlabeled and
        will only query for unlabeled indices.

        Args:
            indices (list): list of ints that are indices

        TODO:
            make generalizable when bounding boxes added to api
        """
        # Check to make sure not already labeled
        unlabeled_indices = list(self.unlabeled_indices & set(indices))
        # Ask for new labels
        new_data = self.problem.get_more_labels(self._indices_to_fnames(unlabeled_indices), self.name)
        # Now we need to infer the column order since it turns out that it can switch...randomly.
        columns = ["id", "class"]
        if has_file_allowed_extension(new_data[0][1]):
            columns = ["class", "id"]
        new_data = pd.DataFrame(new_data, columns=columns)
        # Parse labels and filenames
        n = self.update_targets(new_data, requested=unlabeled_indices)
        print(
            f"Added {n} more labels to the dataset: {self.labeled_size} "
            f"files now labeled, {self.unlabeled_size} unlabeled "
        )

    def get_seed_labels(self, seed_labels: Optional[pd.DataFrame]=None, num_seed_calls: int=0) -> None:
        """
        Get the seed labels from JPL and add them to the dataset.

        Args:
            seed_labels: Seed labels to add or none if want to go get them
            from problem
            num_seed_calls: number of seed labeled level (either 0 or 1)
                necessitated by the secondary_seed_labels in the second checkpoint
                which is considered "seed" labels"

        This also initializes the Categories based on the seed labels.

        """
        if seed_labels is None:
            seed_labels = pd.DataFrame(self.problem.get_seed_labels(self.name, num_seed_calls))

        cat_labels = seed_labels["class"].tolist()

        self.initialize_categories(cat_labels)
        n = self.update_targets(seed_labels)

        print(
            f"Added {n} seed labels to the dataset: {self.labeled_size} "
            f"files now labeled, {self.unlabeled_size} unlabeled "
        )

    def _category_name_to_category_index(self, category_names: List[str]) -> List[int]:
        """
        Given category names, return category indices.

        Args:
            category_names (list[str]): category names

        Returns:
            list[int]: category indices
                corresponding to the input category names.
        """
        return [self.category_to_category_index[cat] for cat in category_names]

    def _category_index_to_category_name(self, category_indices: List[int]) -> List[str]:
        """
        Given category indices, return category names.

        Args:
            category indices (list[int]): category indices

        Returns:
            list[str]: category names
                corresponding to the input category indices
        """
        return [self.category_index_to_category[i] for i in category_indices]

    def update_targets(self, new_labels: Any, requested: List[int]=None, check_redundant: bool=False) -> int:
        """
        Update with new labels for targets.

        Args:
            new_labels (pandas.DataFrame): new labels to add
            requested (list[int]): list of requested labels
        """

        if requested is None:
            requested = []

        n = len(new_labels)

        fnames = new_labels["id"].tolist()
        indices = self._fnames_to_indices(fnames)
        print("nidices", len(indices))

        cat_labels = new_labels["class"].tolist()
        cat_labels = self._category_name_to_category_index(cat_labels)

        # Update labels on images
        unique_images = set(indices + requested)
        num_labeled = 0

        for it, ind in enumerate(indices):
            self.targets[ind] = cat_labels[it]
            num_labeled += 1

        # Update Ids
        self.labeled_indices.update(unique_images)
        self.unlabeled_indices -= unique_images

        self.labeled_size = len(self.labeled_indices)
        self.unlabeled_size = len(self.unlabeled_indices)

        if num_labeled != n:
            warnings.warn(f"{num_labeled}/{n} labels added!  Some already labeled", UserWarning)

        return num_labeled

    def initialize_categories(self, seed_labels: List[str]) -> None:
        """
        Given the seed labels, initialize the category names and indices.

        Args:
            seed_labels (list[str]): list of seed category names
        """
        self.categories = np.unique(seed_labels)
        self.category_to_category_index = dict(zip(self.categories, np.arange(len(self.categories))))
        self.category_index_to_category = dict(zip(np.arange(len(self.categories)), self.categories))

    def _fnames_to_indices(self, fnames: List[str]) -> List[int]:
        """
        Given filenames, return indices.

        Args:
            fnames (list[str]): filenames of images

        Returns:
            list (int):  indices
                indices corresponding to the filenames.
        """
        return [self.image_fname_to_index[fname] for fname in fnames]

    def _indices_to_fnames(self, indices: List[int]) -> List[str]:
        """
        Given indices, return filenames.

        Args:
            indices (list[int]): indices of images

        Returns:
            list (str):  filenames
                corresponding to the indices.
        """
        return [self.index_to_image_fname[i] for i in indices]

    def extra_repr(self) -> str:
        """
        Add extra bit when printing out dataset.

        Returns:
            str: Extra Info of Dataset
        """
        return (
            f"Number of Unlabeled Datapoints {self.unlabeled_size}\n"
            f"Number of Labeled Datapoints {self.labeled_size}"
        )

    def show_example(self, index: int=0) -> None:
        """
        Given an index, show an example.

        Only works for image classification.

        Args:
            index (int, optional, default=0): index of the image you want to show
        """
        out = self[index]
        img = out[0].numpy().transpose(1, 2, 0)
        import matplotlib.pyplot as plt  # type: ignore

        plt.imshow(img)
        plt.title(f"Class: {out[1]}")

    def collate_batch(self, batch: List[Any]) -> Any:
        """
        Collate a batch.

        Custom collate batch function which handles lists and tuples a bit
        differently to accommodate bboxes.  Check out
        https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        for more info.

        Args:
            batch: pytorch batch

        Returns:
            collatated batch

        """
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
            elem = batch[0]
            if elem_type.__name__ == "ndarray":
                return self.collate_batch([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(*(self.collate_batch(samples) for samples in zip(*batch)))
        elif isinstance(elem, tuple):
            return elem_type(self.collate_batch(samples) for samples in zip(*batch))
        elif isinstance(elem, list):  # Slight change to not cluster bboxes!!
            return batch

        raise NotImplementedError

    def dummy_data(self, task_type: str) -> Tuple[Union[Tuple[List[str], List[float], Any], Any], List[int]]:
        """
        Create dummy data for evaluation.

        Args:
            task_type (str): either image_classification or object_detection
            test_imgs (list[str]): list of image names to create fake data for

        Returns:
            preds, indices (same as inference)
        """
        classes = np.random.randint(0, len(self.categories), len(self.image_fnames))
        if task_type == "image_classification":
            return classes, list(self.indices)
        elif task_type == "object_detection":
            bbox = ["20, 20, 80, 80" for _ in range(len(self.image_fnames))]
            conf = [0.95 for _ in range(len(self.image_fnames))]

            preds = (bbox, conf, classes)

            return preds, list(self.indices)
        else:
            raise NotImplementedError

    def format_predictions(self, predictions: Union[Tuple[List, List, List], List], indices: List[int]) -> dict:
        """
        Submit the prediction to JPL vial LwLL class.

        Args:
            predictions (tuple(list,list,list)|list):
                list of prediction as int of class or
                tuple for object detection (bbox, confidence, class)
            indices (list[int]): list of integer indices corresponding to the
                predictions.
        """
        fnames = self._indices_to_fnames(indices)
        preds = self._category_index_to_category_name(predictions)
        df = pd.DataFrame({"id": fnames, "class": preds})

        # Enforce that the labels are strings
        df["class"] = df["class"].astype(str)

        return df.to_dict()


class ObjectDetectionDataset(ImageClassificationDataset):
    """TODO: Define all attributes."""

    def __init__(
        self,
        problem: Any,
        dataset_name: str,
        dataset_root: str,
        transform: Any=ub.NoParam,
        target_transform: Optional[Callable]=None,
        categories: Optional[List[str]]=None,
        seed_labels: Optional[List[str]]=None,
    ) -> None:
        """Initialize."""

        super(ObjectDetectionDataset, self).__init__(
            problem, dataset_name, dataset_root, transform, target_transform, categories, seed_labels,
        )

    def get_more_labels(self, indices: List[int]) -> None:
        """
        Ask LwLL class to interface with JPL server to query for indices.

        This function will check to make sure all requested images are unlabeled and
        will only query for unlabeled indices.

        Args:
            indices (list): list of ints that are indices

        Warning:
            If no labels comes back, this will just assume that there are no labels
                for that image but will mark it as a labeled image.  This is true for
                Object Detection and perhaps some image classification problems

        TODO:
            make generalizable when bounding boxes added to api
        """
        # Check to make sure not already labeled
        unlabeled_indices = list(self.unlabeled_indices & set(indices))
        # Ask for new labels
        new_data = self.problem.get_more_labels(self._indices_to_fnames(unlabeled_indices), self.name)

        columns = ["id", "bbox", "class"]

        new_data = pd.DataFrame(new_data, columns=columns)
        # Parse labels and filenames
        n = self.update_targets(new_data, requested=unlabeled_indices)
        print(
            f"Added {n} seed labels to the dataset: {self.labeled_size} "
            f"files now labeled, {self.unlabeled_size} unlabeled "
        )

    def update_targets(self, new_labels: Any, requested: Optional[List[int]]=None, check_redundant: bool=False) -> int:
        """
        Update with new labels for targets.

        Args:
            new_labels (pandas.DataFrame): new labels to add
            requested (list[int]): list of requested labels
            check_redundant (bool): Whether to check if incoming labels are
                redundant
        """

        if requested is None:
            requested = []

        n = len(new_labels)
        fnames = new_labels["id"].tolist()
        indices = self._fnames_to_indices(fnames)

        cat_labels = new_labels["class"].tolist()
        cat_labels = self._category_name_to_category_index(cat_labels)

        bbox_labels = new_labels["bbox"].tolist()

        # Update labels on images
        unique_images = set(indices + requested)
        num_labeled = 0

        # Either create a new list if no labels, or add it to the previous list
        for it, ind in enumerate(indices):
            new_lab = {
                "category": cat_labels[it],
                "bbox": torch.tensor(list(map(float, bbox_labels[it].split(", ")))),
            }
            if self.targets[ind] is None:
                self.targets[ind] = [new_lab]
                num_labeled += 1
            else:
                # Check if redundant, don't if it so don't add
                unique = True
                if check_redundant:
                    for t in self.targets[ind]:
                        if new_lab["category"] == t["category"] and all(new_lab["bbox"] == t["bbox"]):
                            unique = False
                if unique:
                    self.targets[ind].append(new_lab)
                    num_labeled += 1

        # Update Ids
        self.labeled_indices.update(unique_images)
        self.unlabeled_indices -= unique_images

        self.labeled_size = len(self.labeled_indices)
        self.unlabeled_size = len(self.unlabeled_indices)

        if num_labeled != n:
            warnings.warn(f"{num_labeled}/{n} labels added!  Some already labeled", UserWarning)

        return num_labeled

    def format_predictions(self, predictions: Union[Tuple[List, List, List], List], indices: List[int]) -> dict:
        """
        Submit the prediction to JPL vial LwLL class.

        Args:
            predictions (tuple(list,list,list)|list):
                list of prediction as int of class or
                tuple for object detection (bbox, confidence, class)
            indices (list[int]): list of integer indices corresponding to the
                predictions.
        """
        fnames = self._indices_to_fnames(indices)
        if not isinstance(predictions, tuple):
            raise TypeError("Prediction needs to be tuple for object detection")
        bbox = predictions[0]
        confidence = predictions[1]
        classes = predictions[2]

        classes = self._category_index_to_category_name(classes)
        df = pd.DataFrame({"id": fnames, "bbox": bbox, "confidence": confidence, "class": classes})

        # Enforce that the labels are strings
        df["class"] = df["class"].astype(str)

        return df.to_dict()
