import re


def is_fraction(value: float) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError("Value must be between 0.0 and 1.0")
    return value


def is_date_format(date_str: str) -> str:
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise ValueError("Date must be in 'YYYY-MM-DD' format")
    return date_str
