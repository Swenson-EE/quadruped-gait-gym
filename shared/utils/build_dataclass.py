from dataclasses import fields, is_dataclass, MISSING
from typing import Type, TypeVar, Any, Dict

T = TypeVar("T")


def build_from_flat(
    cls: Type[T],
    flat: Dict[str, Any],
    defaults: Dict[str, Any] | None = None,
    prefix: str = ""
) -> T:
    if defaults is None:
        defaults = {}

    kwargs = {}

    for f in fields(cls):
        field_name = f.name
        field_type = f.type
        full_key = f"{prefix}{field_name}"

        if is_dataclass(field_type):
            # Recurse into nested dataclass
            kwargs[field_name] = build_from_flat(
                field_type,
                flat,
                defaults,
                prefix=full_key + "."
            )
        else:
            value = MISSING

            # Priority 1: flat dict
            if full_key in flat:
                value = flat[full_key]

            # Priority 2: defaults dict
            elif full_key in defaults:
                value = defaults[full_key]

            # Priority 3: dataclass default
            elif f.default is not MISSING:
                value = f.default

            elif f.default_factory is not MISSING:  # type: ignore
                value = f.default_factory()  # type: ignore

            else:
                raise ValueError(f"Missing required field: {full_key}")

            # Optional rounding
            if isinstance(value, float):
                value = round(value, 6)

            kwargs[field_name] = value

    return cls(**kwargs)
