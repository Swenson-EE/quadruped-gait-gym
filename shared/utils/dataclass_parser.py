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


def build_parser_from_dataclass(dc, parser = None, defaults: dict | None = None):
    assert is_dataclass(dc)

    defaults = defaults or {}
    parser = parser or argparse.ArgumentParser()

    for f in fields(dc):
        arg_name = f"--{f.name}"

        field_type = f.type

        # -----------------------------
        # Resolve default priority
        # defaults dict > dataclass default > factory > None
        # -----------------------------
        if f.name in defaults:
            default = defaults[f.name]
        elif f.default_factory is not MISSING:  # type: ignore
            default = f.default_factory()
        elif f.default is not MISSING:
            default = f.default
        else:
            default = None

        kwargs = {"default": default}

        # -----------------------------
        # BOOL
        # -----------------------------
        if field_type == bool:
            kwargs["type"] = str2bool
            kwargs["help"] = f"(boolean) default={default}"

        # -----------------------------
        # ENUM
        # -----------------------------
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            enum_cls = field_type

            # Convert default string -> Enum if needed
            if isinstance(default, str):
                default = enum_cls(default)

            kwargs["type"] = enum_cls
            kwargs["choices"] = list(enum_cls)
            kwargs["default"] = default
            kwargs["help"] = f"(enum) choices={[e.name for e in enum_cls]} default={default.name}"

        # -----------------------------
        # LIST
        # -----------------------------
        elif getattr(field_type, "__origin__", None) == list:
            inner_type = int
            if hasattr(field_type, "__args__") and len(field_type.__args__) == 1:
                inner_type = field_type.__args__[0]

            kwargs["type"] = inner_type
            kwargs["nargs"] = "+"
            kwargs["help"] = f"(list) default={default}"

        # -----------------------------
        # STANDARD TYPES
        # -----------------------------
        else:
            kwargs["type"] = field_type
            kwargs["help"] = f"default={default}"

        parser.add_argument(arg_name, **kwargs)

    return parser


def parse_args_to_dataclass(args, dc_type):
    #args = parser.parse_args()
    args_dict = vars(args)

    final_kwargs = {}

    for f in fields(dc_type):
        value = args_dict.get(f.name, None)

        if value is not None:
            # Convert to Enum if needed
            if isinstance(f.type, type) and issubclass(f.type, Enum):
                if not isinstance(value, f.type):
                    value = f.type(value)

            final_kwargs[f.name] = value

        elif f.default_factory is not MISSING:  # type: ignore
            final_kwargs[f.name] = f.default_factory()

        elif f.default is not MISSING:
            final_kwargs[f.name] = f.default

        else:
            final_kwargs[f.name] = None

    return dc_type(**final_kwargs)
