"""Access package-local chip descriptions and fitted resource prices."""
from importlib.resources import files
import json


def _data_file(kind, filename):
    return files("operator_model").joinpath("data").joinpath(kind).joinpath(filename)


def load_parameters(name):
    return json.loads(_data_file("parameters", name + ".json").read_text(encoding="utf-8"))


def chip_file(name):
    return _data_file("chips", name + ".json")
