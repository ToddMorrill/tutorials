from utils import is_palindrome


def test_is_palindrome_empty_string():
    assert is_palindrome("")


def test_is_palindrome_single_character():
    assert is_palindrome("a")


def test_is_palindrome_mixed_casing():
    assert is_palindrome("Bob")


def test_is_palindrome_with_spaces():
    assert is_palindrome("Never odd or even")


def test_is_palindrome_with_punctuation():
    assert is_palindrome("Do geese see God?")


def test_is_palindrome_not_palindrome():
    assert not is_palindrome("abc")


def test_is_palindrome_not_quite():
    assert not is_palindrome("abab")