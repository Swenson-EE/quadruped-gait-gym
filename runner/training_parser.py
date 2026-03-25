import argparse
from dataclasses import fields, is_dataclass
from enum import Enum


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser_from_dataclass(dc):
    assert is_dataclass(dc)

    parser = argparse.ArgumentParser()

    for f in fields(dc):
        arg_name = f"--{f.name}"
        default = f.default
        field_type = f.type

        kwargs = {"default": default}
        # Handle booleans with store_true / store_false
        if field_type == bool:
            kwargs["type"] = str2bool
            kwargs["help"] = f"(boolean) default={default}"

        # Handle Enums
        elif isinstance(default, Enum) or (
            isinstance(field_type, type) and issubclass(field_type, Enum)
        ):
            enum_cls = field_type
            kwargs["type"] = enum_cls
            kwargs["choices"] = list(enum_cls)

        # Handle List[int] specifically
        elif getattr(field_type, "__origin__", None) == list:
            # assume List[int]
            kwargs["type"] = int
            kwargs["nargs"] = "+"  # parse space-separated integers

        else:
            kwargs["type"] = field_type

        parser.add_argument(arg_name, **kwargs)

    return parser

def parse_args_to_dataclass(parser, dc_type):
    args = parser.parse_args()
    return dc_type(**vars(args))