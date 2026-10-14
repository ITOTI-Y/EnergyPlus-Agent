import pytest

from src.modeling.loads import require_level


def test_the_field_the_method_reads_must_be_given():
    require_level("Watts/Area", watts_per_floor_area=10.0, design_level=None)

    with pytest.raises(ValueError, match="watts_per_person"):
        require_level("Watts/Person", watts_per_floor_area=10.0)
