def roman_to_int(roman: str) -> int:
    roman_map = {
        "i": 1,
        "v": 5,
        "x": 10,
        "l": 50,
        "c": 100,
        "d": 500,
        "m": 1000,
    }
    prev = 0
    total = 0
    for char in reversed(roman.lower()):
        value = roman_map[char]
        if value < prev:
            total -= value
        else:
            total += value
        prev = value
    return total
