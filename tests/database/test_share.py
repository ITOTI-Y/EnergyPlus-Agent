from src._share import UNSET
from src.database.datatools import designday


def test_database_and_converter_share_the_same_unset_sentinel() -> None:
    assert designday.UNSET is UNSET
