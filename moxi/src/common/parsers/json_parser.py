import json
from moxi.src.common.interfaces.types import MoxiNetworkConfiguration


class JosnParser:
    def __init__(self):
        pass

    def parse(self, file_path: str) -> MoxiNetworkConfiguration:
        """
        Function that takes file path to config file and returns processed config object
        """
        try:
            with open(file_path, "r") as file:
                tst_config = json.load(file)

            return tst_config

        except Exception as E:
            print("Could not load configuration file!")
