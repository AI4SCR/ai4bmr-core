import re


def tidy_name(name: str, to_lower: bool = True, split_camel_case: bool = True) -> str:
    """
    Tidy a string by removing whitespace, special characters, and converting to lowercase.

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
    if split_camel_case:
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)

    # 4. Replace special symbols with underscores
    name = re.sub(r"[^\w\s]", "_", name)

    # 5. Convert to lowercase
    if to_lower:
        name = name.lower()

    # 6. Remove multiple underscores in a row (if any), and leading/trailing underscores
    name = re.sub(r"_+", "_", name).strip("_")

    return name

def relabel_duplicates(data, col_name: str):
    cumcounts = data.groupby(col_name).cumcount()
    data.loc[:, col_name] = data[col_name] + "_" + cumcounts.astype(str)
    data.loc[:, col_name] = data[col_name].str.replace("_0$", "", regex=True)
    return data