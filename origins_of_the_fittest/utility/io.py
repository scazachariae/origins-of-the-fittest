import pyarrow as pa
import importlib.resources as ir
import polars as pl
import pandas as pd
import os
import re


def check_existance(filename: str) -> bool:
    """
    Check if a file exists in the package's data directory.

    Args:
        filename (str): Name of the file to look up.

    Returns:
        bool: True if the file exists, False otherwise.
    """

    package_data = "origins_of_the_fittest.data"
    try:
        path = ir.files(package_data) / filename
        return path.is_file()
    except Exception:
        return False


def get_available_seeds(template: str) -> list[int]:
    """
    Scan the package's data resources for files matching the given template
    and return all integer seed values found.

    Args:
        template (str): Filename pattern with exactly one '{seed}' placeholder,
            e.g. 'my_graph_seed{seed}.parquet'.

    Returns:
        list[int]: Sorted list of seed values extracted from matching filenames.
    """
    package_data = "origins_of_the_fittest.data"
    # Build a regex: escape the template except for '{seed}'
    escaped = re.escape(template).replace(r"\{seed\}", r"(\d+)")
    pattern = re.compile(f"^{escaped}$")

    seeds: list[int] = []
    for entry in ir.files(package_data).iterdir():
        fname = entry.name
        match = pattern.match(fname)
        if match:
            seeds.append(int(match.group(1)))

    return sorted(seeds)


def load_data_pd(filename: str) -> pd.DataFrame:
    """
    Load a Parquet file from the package's bundled data directory.

    Args:
        filename (str): Name of the parquet file to read.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    # Determine the file extension
    _, ext = os.path.splitext(filename)

    package_data = "origins_of_the_fittest.data"

    path = ir.files(package_data) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"The file '{filename}' was not found in the data directory."
        )

    with path.open("rb") as file:
        if ext.lower() == ".parquet":
            return pd.read_parquet(file)
        else:
            raise NotImplementedError(f"File format '{ext}' is not supported yet.")


def load_network(filename: str) -> pd.DataFrame:
    """
    Load an edge-list network from the package's data directory.

    Args:
        filename (str): Name of the edge-list file to load.

    Returns:
        pd.DataFrame: Symmetric adjacency matrix stored in a pandas DataFrame.
    """

    nw = load_data_pd(filename).assign(weight=1)
    nw = pd.concat(
        [
            nw,
            nw.rename(columns={"source": "destination", "destination": "source"}),
        ]
    )
    nw = nw.pivot(index="source", columns="destination", values="weight").fillna(0)

    return nw


def load_data_pl(filename: str) -> pl.DataFrame:
    """
    Load a Parquet file from the package's bundled data directory using Polars.

    Args:
        filename (str): Name of the parquet file to read.

    Returns:
        pl.DataFrame: Loaded data as a Polars DataFrame.
    """
    # Determine the file extension
    _, ext = os.path.splitext(filename)

    package_data = "origins_of_the_fittest.data"

    path = ir.files(package_data) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"The file '{filename}' was not found in the data directory."
        )

    with path.open("rb") as file:
        if ext.lower() == ".parquet":
            return pl.read_parquet(file)
        else:
            raise NotImplementedError(f"File format '{ext}' is not supported yet.")


def write_arrow(df, filename):
    df_copy = df.copy()

    for col in df_copy.columns:
        if df_copy[col].dtype == "int64":
            df_copy[col] = df_copy[col].astype("int32")
        if df_copy[col].dtype == "float64":
            df_copy[col] = df_copy[col].astype("float32")
        if df_copy[col].dtype == "<M8[ns]":
            df_copy[col] = df_copy[col].astype("string")

    fd = open(filename, "wb")
    batch = pa.RecordBatch.from_pandas(df_copy, preserve_index=False)
    writer = pa.ipc.RecordBatchStreamWriter(fd, batch.schema)
    writer.write_batch(batch)
    writer.close()
    fd.close()

    print(batch)
