import re
import sys
import pathlib
import requests
from rich import progress as rich_progress
import zipfile

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

if not DATA_DIR.exists():
    if DATA_DIR.parent.name != "paper":
        print(
            f"cannot find the data directory {DATA_DIR}, this file belongs in the `paper/nbks` directory"
        )
        sys.exit(1)
    else:
        DATA_DIR.mkdir()

_filename = re.compile(r"/([^/]+\.[^/?]+)(?:\?|$)")


def extract_zip(zip_file_path, extract_dir):
    """
    Extracts the contents of a ZIP archive to a specified directory.

    Args:
        zip_file_path (str): Path to the ZIP archive file.
        extract_dir (str): Directory to extract the contents to.
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def fetch_data_sets():
    urls = [
        # the following two are required for checking the dvs prep command
        # "https://zenodo.org/records/14052787/files/refsoil.zip?download=1",
        # "https://zenodo.org/records/14052787/files/wol.zip?download=1",
        # the following is required for the tree-based comparisons
        "https://zenodo.org/records/14052787/files/mammals-aligned.zip?download=1",
        # the following files are required for checking performance against large
        # data set sizes. They are around(1GB and 9GB) and can take some time to download!
        # you can grab them directly from zenodo and place into the paper/data directory
        # "https://zenodo.org/records/14052787/files/soil.dvseqs?download=1",
        # "https://zenodo.org/records/14052787/files/wol.dvseqs?download=1",
    ]
    block_size = 4096

    with rich_progress.Progress() as progress:
        get_url = progress.add_task("Downloading data sets", total=len(urls))
        for url in urls:
            name = _filename.search(url).group(1)
            dest = DATA_DIR / name
            response = requests.get(url, stream=True)

            total_size = int(response.headers.get("content-length", 0))
            get_content = progress.add_task(f"Getting {name}", total=total_size)

            with dest.open("wb") as out:
                for data in response.iter_content(block_size):
                    progress.update(get_content, advance=len(data), refresh=True)
                    out.write(data)
            progress.remove_task(get_content)
            progress.update(get_url, advance=1, refresh=True)
            if name.endswith(".zip"):
                extract_zip(dest, DATA_DIR)


fetch_data_sets()
