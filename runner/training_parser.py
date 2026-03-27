import argparse
from dataclasses import fields, is_dataclass, MISSING
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

        # Use default_factory if it exists, else default, else None
        if f.default_factory is not MISSING:  # type: ignore
            default = f.default_factory()
        elif f.default is not MISSING:
            default = f.default
        else:
            default = None

        field_type = f.type
        kwargs = {"default": default}

        # Handle booleans with str2bool
        if field_type == bool:
            kwargs["type"] = str2bool
            kwargs["help"] = f"(boolean) default={default}"

        # Handle Enums
        elif isinstance(default, Enum) or (isinstance(field_type, type) and issubclass(field_type, Enum)):
            enum_cls = field_type
            kwargs["type"] = enum_cls
            kwargs["choices"] = list(enum_cls)
            kwargs["help"] = f"(enum) choices={[e.name for e in enum_cls]} default={default.name}"

        # Handle List[int] (or generally List[some_type])
        elif getattr(field_type, "__origin__", None) == list:
            # Try to infer inner type
            inner_type = int  # default to int if unspecified
            if hasattr(field_type, "__args__") and len(field_type.__args__) == 1:
                inner_type = field_type.__args__[0]
            kwargs["type"] = inner_type
            kwargs["nargs"] = "+"
            kwargs["help"] = f"(list) default={default}"

        else:
            kwargs["type"] = field_type
            kwargs["help"] = f"default={default}"

        parser.add_argument(arg_name, **kwargs)

    return parser


def parse_args_to_dataclass(parser, dc_type):
    args = parser.parse_args()
    args_dict = vars(args)

    # Ensure any missing values use dataclass defaults
    final_kwargs = {}
    for f in fields(dc_type):
        if f.name in args_dict and args_dict[f.name] is not None:
            final_kwargs[f.name] = args_dict[f.name]
        elif f.default_factory is not MISSING:  # type: ignore
            final_kwargs[f.name] = f.default_factory()
        elif f.default is not MISSING:
            final_kwargs[f.name] = f.default
        else:
            final_kwargs[f.name] = None

    return dc_type(**final_kwargs)
