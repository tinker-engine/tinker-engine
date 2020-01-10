from pathlib import Path
import fire
import boto3
import sys
import os
import tarfile
import shutil


class ProgressPercentage(object):
    def __init__(self, o_s3bucket, key_name: str) -> None:  # type: ignore
        self._key_name = key_name
        boto_client = o_s3bucket.meta.client
        # ContentLength is an int
        self._size = boto_client.head_object(Bucket=o_s3bucket.name, Key=key_name)['ContentLength']
        self._seen_so_far = 0
        sys.stdout.write('\n')

    def __call__(self, bytes_amount: int) -> None:
        self._seen_so_far += bytes_amount
        percentage = (float(self._seen_so_far) / float(self._size)) * 100
        TERM_UP_ONE_LINE = '\033[A'
        TERM_CLEAR_LINE = '\033[2K'
        sys.stdout.write('\r' + TERM_UP_ONE_LINE + TERM_CLEAR_LINE)
        sys.stdout.write('{} {}/{} ({}%)\n'.format(self._key_name, str(self._seen_so_far), str(self._size), str(percentage)))
        sys.stdout.flush()


class CLI(object):
    """
    This utility provides easy access and download of the compressed datasets that
    are put into the proper LwLL form

   To ensure the script works correctly, execute the following:
        - sudo mkdir /datasets
        - sudo chmod 777 /datasets
    """

    def __init__(self) -> None:
        self.session = boto3.Session(
            aws_access_key_id='AKIAXNTA46J3YJ6LRKO7',
            aws_secret_access_key='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',
            region_name='us-east-1',
        )
        self.bucket_name = 'lwll-datasets'
        self.compressed_data_path = 'compressed_datasets/'
        self.client = self.session.client('s3')
        self.bucket = self.session.resource('s3').Bucket(self.bucket_name)

    def download_data(self, dataset: str, overwrite: bool = False,
                      dataset_dir: str = '/datasets/lwll_datasets') -> str:
        """
        Utility to method to download and unzip the compressed datasets from our S3
        bucket

        Args
            dataset : str
                the dataset name, or `all`, which will go through and download all
                datasets one by one.
            overwrite_problem : bool
                determines whether or not to do an overwrite the dataset location
                locally. If `True`, and a directory exists with the name already, we
                will not attempt to download and unzip.
            dataset_dir : str
                directory to put the datasets in.
        """
        datasets = self._list_data()

        # Validate we have a valid dataset name passed in
        if dataset != 'all' and dataset not in datasets:
            print('Invalid dataset')
            sys.exit()

        # Download and unzip the datasets
        if dataset == 'all':
            for _dataset in datasets:
                self._download_dataset(_dataset, overwrite, dataset_dir)
        else:
            self._download_dataset(dataset, overwrite, dataset_dir)

        return 'Done!'

    def list_data(self) -> list:
        """
        Utility method to list all available datasets currently processed.
        This is to list what datasets are available for download if you
            only want to download a subset
        """

        keys = self._list_data(verbose=True)
        return keys

    def _list_data(self, verbose: bool = False) -> list:

        contents = self.client.list_objects(Bucket=self.bucket_name, Prefix=self.compressed_data_path, Delimiter='/')['Contents']
        keys = [d['Key'].split('/')[-1].split('.')[0] for d in contents]
        if verbose:
            print("Available Datasets:")
        return keys

    def _download_dataset(self, dataset: str, overwrite: bool, dataset_dir: str) -> None:
        if Path(f'{dataset_dir}/{dataset}').is_dir() and not overwrite:
            print(f"{dataset} is already downloaded and `overwrite` is set to False. \
                    Not downloading `{dataset}`. *Note: This does not guaruntee the newest \
                    version of the dataset...")
        else:
            if Path(f'{dataset_dir}/{dataset}').exists():
                shutil.rmtree(f'{dataset_dir}/{dataset}')
            if not Path(f'{dataset_dir}').exists():
                os.makedirs(f'{dataset_dir}')
            # Download
            progress = ProgressPercentage(self.bucket, f'{self.compressed_data_path}{dataset}.tar.gz')
            self.bucket.download_file(f'{self.compressed_data_path}{dataset}.tar.gz', f'{dataset_dir}/{dataset}.tar.gz', Callback=progress)
            # Extract
            tarfile.open(f'{dataset_dir}/{dataset}.tar.gz', 'r:gz').extractall(f'{dataset_dir}')
            # Remove Zip
            os.remove(f'{dataset_dir}/{dataset}.tar.gz')
        return


def main() -> None:
    fire.Fire(CLI)


if __name__ == '__main__':
    fire.Fire(CLI)
