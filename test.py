from main import add


def test_add():
    assert add(5, 6) == 11


def test_wrong_add():
    assert add(3, 4) == 7
