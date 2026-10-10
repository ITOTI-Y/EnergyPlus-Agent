"""Create, read, update and delete named objects with reference checks.

idfpy accepts references to objects that do not exist yet and only reports
them from ``IDF.validate()``; these operations reject them at the call
instead, so a tool caller learns about a typo when it makes it.
"""

from typing import Any, Final

from idfpy import IDF, IDFBaseModel, RefError

from src.modeling.errors import (
    DuplicateNameError,
    MissingReferenceError,
    ObjectNotFoundError,
    ReferencedObjectError,
)


def label(obj: IDFBaseModel) -> str:
    """``Type:name`` of an object, or its type alone when it has no name."""
    name = getattr(obj, "name", None)
    return f"{obj.idf_object_type()}:{name}" if name else obj.idf_object_type()


def missing_references(idf: IDF, obj: IDFBaseModel, name: str) -> list[str]:
    """References of ``obj`` that no object in ``idf`` provides.

    ``obj`` itself need not be in ``idf``. Wraps idfpy's private per-object
    validation, the only check that works before an object is added; update
    this call if idfpy changes it.
    """
    errors: list[RefError] = []
    idf._validate_obj_refs(name, obj, errors)
    return [f"{err.field_name}: {err.detail}" for err in errors]


PLACEHOLDER_NAMES: Final = frozenset({"none", "null", "undefined", "nan"})


def create[T: IDFBaseModel](idf: IDF, obj: T) -> T:
    """Add ``obj`` after checking its name is free and its references exist.

    Raises:
        ValueError: If the name is a placeholder such as 'None'.
        DuplicateNameError: If an object of the same type has the name.
        MissingReferenceError: If a reference names no existing object.
    """
    object_type = obj.idf_object_type()
    name = getattr(obj, "name", None) or object_type
    if name.strip().lower() in PLACEHOLDER_NAMES:
        # A model that lost track of a name sends the string of a null.
        raise ValueError(f"'{name}' is not a name; give the {object_type} a real one.")
    if hasattr(obj, "name") and idf.has(type(obj), name):
        raise DuplicateNameError(object_type, name)
    if missing := missing_references(idf, obj, name):
        raise MissingReferenceError(object_type, name, missing)
    idf.add(obj)
    return obj


def create_or_same[T: IDFBaseModel](idf: IDF, obj: T) -> tuple[T, bool]:
    """Like ``create``, but an equal object already present is accepted.

    Reference data repeats objects: two prototype constructions share a
    layer, and a phase that creates each construction's materials meets the
    same material twice.

    Returns:
        The object in the model, and whether it was created now.

    Raises:
        DuplicateNameError: If an object of the same name differs.
        MissingReferenceError: If a reference names no existing object.
    """
    name = getattr(obj, "name", None)
    existing = idf.get(type(obj), name) if name else None
    if existing is not None and existing.model_dump() == obj.model_dump():
        return existing, False
    return create(idf, obj), True


def get[T: IDFBaseModel](idf: IDF, object_type: type[T], name: str) -> T:
    """Raises: ObjectNotFoundError: If no such object exists."""
    obj = idf.get(object_type, name)
    if obj is None:
        raise ObjectNotFoundError(object_type.idf_object_type(), name)
    return obj


def update[T: IDFBaseModel](idf: IDF, obj: T, changes: dict[str, Any]) -> T:
    """Apply field changes all at once; a ``name`` change renames references.

    The changed object is validated as a whole before any field is assigned,
    so a rejected update leaves the model untouched.

    Raises:
        pydantic.ValidationError: If a changed value is invalid.
        DuplicateNameError: If the new name is taken.
        MissingReferenceError: If a changed reference names no existing object.
    """
    object_type = type(obj)
    name = getattr(obj, "name", None) or label(obj)
    candidate = object_type.model_validate(
        {**obj.model_dump(exclude_unset=True), **changes}
    )
    new_name = changes.get("name", name)
    if new_name != name and idf.has(object_type, new_name):
        raise DuplicateNameError(obj.idf_object_type(), new_name)
    if missing := missing_references(idf, candidate, name):
        raise MissingReferenceError(obj.idf_object_type(), name, missing)
    for field in changes:
        # Assigning a provider name makes idfpy rename every reference to it.
        setattr(obj, field, getattr(candidate, field))
    return obj


def delete(idf: IDF, obj: IDFBaseModel, key: str) -> None:
    """Remove ``obj`` stored under ``key`` unless another object references it.

    Raises:
        ReferencedObjectError: If any object references ``obj``.
    """
    if referrers := obj.referencing():
        raise ReferencedObjectError(
            obj.idf_object_type(), key, [label(r) for r in referrers]
        )
    idf.remove(type(obj), key)


def dumps(objects: dict[str, IDFBaseModel]) -> list[dict[str, Any]]:
    return [obj.model_dump(exclude_none=True) for obj in objects.values()]
