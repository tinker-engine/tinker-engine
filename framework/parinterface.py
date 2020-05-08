"""Client implementation for Par interface."""

import requests
import json
from json import JSONDecodeError
import io
import os

from typing import Any, Generator, Optional, Dict
from requests import Response
from uuid import UUID

class ParInterface(Harness):
    def __init__(self, api_url: str, folder: str = ".") -> None:
        """
        Initialize a client connection object.

        :param api_url: url for where server is hosted
        """
        self.api_url = api_url
        self.folder = folder

    def _check_response(response: Response) -> None:
        """
        Raise the appropriate ApiError based on response error code.
    
        :param response:
        :return: True
        """
        if response.status_code != 200:
            #TODO: print more thorough information here
            print( "Server error: ", response.status_code )
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
            -protocol
            -domain
            -detector_seed
            -test_assumptions
        Returns:
            -detection_seed
            -test_ids filename
        """
        payload = {
            "protocol": protocol,
            "domain": domain,
            "detector_seed": detector_seed,
        }

        with open(test_assumptions, "r") as f:
            contents = f.read()

        response = request.get(
            "/test/ids",
            files={
                "test_requirements": io.StringIO(json.dumps(payload)),
                "test_assumptions": io.StringIO(contents),
            },
        )

        _check_response(response)

        header = response.headers["Content-Disposition"]
        header_dict = {
            x[0].strip(): x[1].strip(" \"'")
            for x in [part.split("=") for part in header.split(";") if "=" in part]
        }

        filename = os.path.abspath(os.path.join(self.folder, header_dict["filename"]))
        with open(filename, "w") as f:
            f.write(response.content.decode("utf-8"))

        return filename

    def session_request( self, test_ids: str, protocol: str, novelty_detector_version: str):
        """
        Create a new session to evaluate the detector using an empirical protocol.

        Arguments:
            -test_ids
            -protocol
            -novelty_detector_version
        Returns:
        """
        payload = {
            "protocol": protocol,
            "novelty_detector_version": novelty_detector_version,
        }

        with open(test_ids, "r") as f:
            contents = f.read()

        response = request.post(
            "/session",
            files={
                "test_ids": io.StringIO(contents),
                "configuration": io.StringIO(json.dumps(payload)),
            },
        )

        _check_response(response)
        self.session_id = response.json()["session_id"]

    def dataset_request(self, test_id: UUID, round_id: int) -> str:
        """
        Request data for evaluation.

        Arguments:
            -test_id
            -round_id
        Returns:
            -num_samples
            -dataset_uris filename
        """
        response = request.get(
            "/session/dataset",
            params={"session_id": self.session_id, "test_id": test_id, "round_id": round_id},
        )

        _check_response(response)

        header = response.headers["Content-Disposition"]
        header_dict = {
            x[0].strip(): x[1].strip(" \"'")
            for x in [part.split("=") for part in header.split(";") if "=" in part]
        }
        filename = os.path.abspath(os.path.join(self.folder, header_dict["filename"]))
        with open(filename, "wb") as f:
            f.write(response.content)

        return filename

    def get_feedback_request(
        self,
        feedback_ids: list,
        test_id: str,
        round_id: int,
        feedback_type: str,
    ) -> Dict[str, Any]:
        """
        Get Labels from the server based provided one or more example ids.

        Arguments:
            -feedback_ids
            -test_id
            -round_id
            -feedback_type -- label, detection, characterization
        Returns:
            -labels dictionary
        """
        response = request.get(
            "/session/feedback",
            params={
                "feedback_ids": "|".join(feedback_ids),
                "session_id": self.session_id,
                "test_id": test_id,
                "round_id": round_id,
                "feedback_type": feedback_type,
            },
        )

        _check_response(response)

        return response.json()

    def post_results( self, result_files: Dict[str, str], test_id: UUID, round_id: int,) -> None:
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
            "result_types": "|".join(result_files.keys())
        }

        files = {"test_identification": io.StringIO(json.dumps(payload))}

        if len(result_files.keys()) == 0:
            raise Exception("Must provide at least one result file")

        for r_type in result_files:
            with open(result_files[r_type], "r") as f:
                contents = f.read()
                files[f"{r_type}_file"] = io.StringIO(contents)

        response = request.post("/session/results", files=files)

        _check_response(response)

    def evaluate(self, test_id: str, round_id: int) -> str:
        """
        Get results for test(s).

        Arguments:
            -test_id
            -round_id
        Returns:
            -filename
        """
        response = request.get(
            "/session/evaluations",
            params={"session_id": self.session_id, "test_id": test_id, "round_id": round_id},
        )

        _check_response(response)

        header = response.headers["Content-Disposition"]
        header_dict = {
            x[0].strip(): x[1].strip(" \"'")
            for x in [part.split("=") for part in header.split(";") if "=" in part]
        }

        filename = os.path.abspath(os.path.join(self.folder, header_dict["filename"]))
        with open(filename, "w") as f:
            f.write(response.content.decode("utf-8"))

        return filename

    def terminate_session(self) -> None:
        """
        Terminate the session after the evaluation for the protocol is complete.

        Arguments:
        Returns: No return
        """
        response = request.delete("/session", params={"session_id": self.session_id})

        _check_response(response)

