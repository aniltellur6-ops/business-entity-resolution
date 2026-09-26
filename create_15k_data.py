import os
import pandas as pd


SOURCE_DIR = r"D:\Amazon ML Challenge 2026\data"
OUTPUT_DIR = r"D:\Amazon ML 15K\data"

FILES = [
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv"
]

ROWS = 15000


os.makedirs(OUTPUT_DIR, exist_ok=True)


for filename in FILES:

    input_file = os.path.join(
        SOURCE_DIR,
        filename
    )

    name, extension = os.path.splitext(filename)

    output_file = os.path.join(
        OUTPUT_DIR,
        name + "_15000" + extension
    )

    print("=" * 60)
    print("Processing:", filename)
    print("=" * 60)

    df = pd.read_csv(
        input_file,
        sep="\t",
        nrows=ROWS
    )

    df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print("Rows copied:", len(df))
    print("Output:", output_file)
    print()


print("=" * 60)
print("15K DATA CREATION COMPLETED")
print("=" * 60)