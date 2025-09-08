from typing import Protocol
from dataclasses import dataclass
from abc import abstractmethod
from moxi.src.common.parsers.json_parser import JosnParser


class ParserWorker(Protocol):
    @abstractmethod
    def parse(self, file_path: str) -> dict: ...


@dataclass
class ParserWorkerConfig:
    parser_name: str
    parser: ParserWorker


class ConfigutationParser:
    def __init__(self):
        self.parsers: dict[ParserWorkerConfig] = {}

    def add_parser(self, parser_name: str, parser: ParserWorker) -> None:
        self.parsers[parser_name] = ParserWorkerConfig(
            parser_name=parser_name, parser=parser
        )
        return self

    def parse_config(self, parser_name: str, file_path: str) -> dict:
        return self.parsers[parser_name].parser.parse(file_path)


def configuration_parser_builder(
    parsers: dict[str, ParserWorker],
) -> ConfigutationParser:
    loader_parser = ConfigutationParser()
    for parser_name, parser in parsers.items():
        loader_parser.add_parser(parser_name, parser)
    return loader_parser


def init_parser() -> ConfigutationParser:
    parsers = {"json_parser": JosnParser()}
    config_parser = configuration_parser_builder(parsers)
    return config_parser


def moxi_config_parser(file_path: str):
    parser = init_parser()

    if ".json" in file_path.lower():
        return parser.parse_config("json_parser", file_path)

    # default case
    raise ValueError(f"Unsupported file type: {file_path}")
