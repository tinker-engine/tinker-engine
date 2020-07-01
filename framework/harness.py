"""Test Harness."""
import os
import json
import logging


class Harness:
    """Harness."""

    def __init__(self, json_configuration_file, interface_config_path):
        """Initialize."""

        json_full_path = os.path.join(interface_config_path, json_configuration_file)
        if not os.path.exists(json_full_path):
            logging.error("Given LocalInterface configuration file does not exist")
            exit(1)

        with open(json_full_path) as json_file:
            self.configuration_data = json.load(json_file)
        self.metadata = None
        self.toolset = {}
