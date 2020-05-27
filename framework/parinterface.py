"""Client implementation for Par interface."""

import requests
import json
import io
import os

from framework.harness import Harness
from typing import Any, Dict
from requests import Response
from uuid import UUID


class ParInterface(Harness):
    """Interface to PAR server."""

    def __init__(self, configfile, configfolder) -> None:
        """
        Initialize a client connection object.

        :param api_url: url for where server is hosted
        """
        Harness.__init__(self, configfile, configfolder)
        self.api_url = self.configuration_data["url"]
        self.folder = configfolder

    def _check_response(self, response: Response) -> None:
        """
        Produce appropriate output on error.

        :param response:
        :return: True
        """
        if response.status_code != 200:
            # TODO: print more thorough information here
            print("Server error: ", response.status_code)
            exit(1)

    def test_ids_request(
        self,
        protocol: str,
        domain: str,
        detector_seed: str,
        test_assumptions: str = "{}",
    ) -> str:
        """
        Request Test Identifiers as part of a series of individual tests.

        Arguments:
            -protocol   : string indicating which protocol is being evaluated
            -domain     :
            -detector_seed
            -test_assumptions
        Returns:
            -filename of file containing test ids
        """
        payload = {
            "protocol": protocol,
            "domain": domain,
            "detector_seed": detector_seed,
        }

        with open(test_assumptions, "r") as f:
            contents = f.read()

        response = requests.get(
            f"{self.api_url}/test/ids",
            files={
                "test_requirements": io.StringIO(json.dumps(payload)),
                "test_assumptions": io.StringIO(contents),
            },
        )

        self._check_response(response)

        header = response.headers["Content-Disposition"]
        header_dict = {
            x[0].strip(): x[1].strip(" \"'")
            for x in [part.split("=") for part in header.split(";") if "=" in part]
        }

        filename = os.path.abspath(os.path.join(self.folder, header_dict["filename"]))
        with open(filename, "w") as f:
            f.write(response.content.decode("utf-8"))

        return filename

    def session_request(
        self, test_ids: list, protocol: str, novelty_detector_version: str
    ):
        """
        Create a new session to evaluate the detector using an empirical protocol.

        Arguments:
            -test_ids   : list of tests being evaluated in this session
            -protocol   : string indicating which protocol is being evaluated
            -novelty_detector_version : string indicating the version of the novelty detector being evaluated
        Returns:
            -session id
        """
        payload = {
            "protocol": protocol,
            "novelty_detector_version": novelty_detector_version,
        }

        ids = "\n".join(test_ids) + "\n"

        response = requests.post(
            f"{self.api_url}/session",
            files={"test_ids": ids, "configuration": io.StringIO(json.dumps(payload))},
        )

        self._check_response(response)
        self.session_id = response.json()["session_id"]
        return self.session_id

    def dataset_request(self, test_id: UUID, round_id: int) -> str:
        """
        Request data for evaluation.

        Arguments:
            -test_id    : the test being evaluated at this moment.
            -round_id   : the sequential number of the round being evaluated
        Returns:
            -filename of a file containing a list of image files (including full path for each)
        """
        response = requests.get(
            f"{self.api_url}/session/dataset",
            params={
                "session_id": self.session_id,
                "test_id": test_id,
                "round_id": round_id,
            },
        )

        self._check_response(response)

        header = response.headers["Content-Disposition"]
        header_dict = {
            x[0].strip(): x[1].strip(" \"'")
            for x in [part.split("=") for part in header.split(";") if "=" in part]
        }
        filename = os.path.abspath(os.path.join(self.folder, header_dict["filename"]))
        with open(filename, "wb") as f:
            f.write(response.content)

        new_filename = self._append_data_root_to_dataset(filename, test_id)

        return new_filename

    # TODO: merge this code directly into dataset_request, and stop writing so
    # many files
    def _append_data_root_to_dataset(self, dataset_path: str, test_id: UUID) -> str:
        assert os.path.exists(dataset_path)
        orig_dataset = open(dataset_path, "r")
        with open(dataset_path, "r") as orig_dataset:
            image_names = orig_dataset.readlines()
            image_paths = [
                os.path.join(self.configuration_data["data_location"], image_name)
                for image_name in image_names
            ]
        new_dataset_file = f"{self.session_id}_{test_id}.csv"
        new_dataset_path = os.path.join(os.getcwd(), new_dataset_file)
        with open(new_dataset_path, "w") as new_dataset:
            new_dataset.writelines(image_paths)
        return new_dataset_path

    def get_feedback_request(
        self, feedback_ids: list, feedback_type: str, test_id: str, round_id: int,
    ) -> Dict[str, Any]:
        """
        Get Labels from the server based provided one or more example ids.

        Arguments:
            -feedback_ids
            -test_id        : the id of the test currently being evaluated
            -round_id       : the sequential number of the round being evaluated
            -feedback_type -- label, detection, characterization
        Returns:
            -labels dictionary
        """
        response = requests.get(
            f"{self.api_url}/session/feedback",
            params={
                "feedback_ids": "|".join(feedback_ids),
                "session_id": self.session_id,
                "test_id": test_id,
                "round_id": round_id,
                "feedback_type": feedback_type,
            },
        )

        self._check_response(response)

        return response.json()

    def post_results(
        self, result_files: Dict[str, str], test_id: UUID, round_id: int,
    ) -> None:
        """
        Post client detector predictions for the dataset.

        Arguments:
            -result_files (dict of "type : file")
            -session_id
            -test_id
            -round_id
        Returns: No return
        """
        payload = {
            "session_id": self.session_id,
            "test_id": test_id,
            "round_id": round_id,
            "result_types": "|".join(result_files.keys()),
        }

        files = {"test_identification": io.StringIO(json.dumps(payload))}

        if len(result_files.keys()) == 0:
            raise Exception("Must provide at least one result file")

        for r_type in result_files:
            with open(result_files[r_type], "r") as f:
                contents = f.read()
                files[f"{r_type}_file"] = io.StringIO(contents)

        response = requests.post(f"{self.api_url}/session/results", files=files)

        self._check_response(response)

    def evaluate(self, test_id: str, round_id: int) -> str:
        """
        Get results for test(s).

        Arguments:
            -test_id
            -round_id
        Returns:
            -filename
        """
        response = requests.get(
            f"{self.api_url}/session/evaluations",
            params={
                "session_id": self.session_id,
                "test_id": test_id,
                "round_id": round_id,
            },
        )

        self._check_response(response)

        header = response.headers["Content-Disposition"]
        header_dict = {
            x[0].strip(): x[1].strip(" \"'")
            for x in [part.split("=") for part in header.split(";") if "=" in part]
        }

        filename = os.path.abspath(os.path.join(self.folder, header_dict["filename"]))
        with open(filename, "w") as f:
            f.write(response.content.decode("utf-8"))

        return filename

    def get_test_metadata(self, test_id: str) -> Dict[str, Any]:
        """
        Retrieve the metadata json for the specified test.

        Arguments:
            -test_id
        Returns:
            metadata json
        """
        response = requests.get(
            f"{self.api_url}/test/metadata", params={"test_id": test_id},
        )

        self._check_response(response)
        return response.json()

    def terminate_session(self) -> None:
        """
        Terminate the session after the evaluation for the protocol is complete.

        Arguments:
        Returns: No return
        """
        response = requests.delete(
            f"{self.api_url}/session", params={"session_id": self.session_id}
        )

        self._check_response(response)
        self.session_id = None
