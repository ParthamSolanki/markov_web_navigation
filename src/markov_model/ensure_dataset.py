import gzip
import urllib.request
import zipfile

from .config import SEQ_FILE, SOURCE_URL


# This function checks if the dataset exists locally otherwise it downloads it for usage.
def ensure_msnbc_dataset(url=SOURCE_URL, file_path=SEQ_FILE):
    if file_path.exists():
        print(f"Dataset exists at -'{file_path}'")
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = file_path.parent / "msnbc.zip"
    print(f"Dataset not found at location '{file_path}'. Downloading...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as z:
        # Extract the .seq file directly into the data folder
        gz_file = next(
            f for f in z.namelist() if f.endswith("seq.gz") or f.endswith(".seq")
        )
        gz_data = z.read(gz_file)

    if gz_file.endswith(".gz"):
        raw_seq = gzip.decompress(gz_data)
    else:
        raw_seq = gz_data

    with open(file_path, "wb") as f:
        f.write(raw_seq)

    if zip_path.exists():
        zip_path.unlink()

    print(f"Dataset successfully downloaded to - '{file_path}'.")
