import re


def tidy_names(name: str) -> str:
    """
    Tidy a string to make it a valid name for a variable, function, or class in Python.
    Args:
        name: string to tidy

    Returns:
        str
    """
    # 1. Remove leading and trailing whitespace
    name = name.strip()

    # 2. Replace white spaces with underscores
    name = re.sub(r"\s+", "_", name)

    # 3. Split CamelCase into separate words with underscores
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)

    # 4. Replace special symbols with underscores
    name = re.sub(r"[^\w\s]", "_", name)

    # 5. Convert to lowercase
    name = name.lower()

    # 6. Remove multiple underscores in a row (if any), and leading/trailing underscores
    name = re.sub(r"_+", "_", name).strip("_")

    return name
