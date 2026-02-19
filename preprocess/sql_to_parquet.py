import argparse
from pathlib import Path
import sqlite3

import pandas as pd
from tqdm import tqdm


def sqlite_table_to_parquet(
    sqlite_path: str,
    table_name: str,
    parquet_path: str,
    context_length: int,
    stride: int = 1,
) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        query = (
            f"SELECT * FROM {table_name} "
            f"ORDER BY DATETIME(timestamp) ASC "
        )
        df: pd.DataFrame = pd.read_sql_query(query, conn)
        if df.empty:
            return

        content = df['content']
        timestamp_col = df["timestamp"]
        indices: list[int] = []
        positive_sentences: list[str] = []
        groups: list[list[str]] = []
        timestamps: list[str] = []

        current_loop = tqdm(range(0, content.__len__() - context_length + 1, stride))
        current_loop.set_description(table_name)
        for i in current_loop:
            group = content[i:i+context_length]
            positive: str = ''.join([
                f"<user{j}>{s}</user>"
                for j, s in enumerate(group)
            ])
            indices.append(i)
            groups.append(group.to_list())
            positive_sentences.append(positive)
            # timestamps.append(timestamp_col[i].to_list())
            timestamps.append(timestamp_col[i])

        if positive_sentences.__len__() == 0:
            return

        new_df = pd.DataFrame({
            'index': indices,
            'positive': positive_sentences,
            'group': groups,
            'timestamp': timestamps,
        })

        new_df.to_parquet(
            parquet_path,
            engine="fastparquet",
            index=False,
        )
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="SQLite Database file to load from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory in which to write parquet files.",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=8,
        help="Message context length.",
    )
    parser.add_argument(
        "--table_names",
        type=str,
        default=None,
        nargs='+',
        help="Table names to select.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride parameter of sliding windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_filename: str = args.input_file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_length: int = args.context_len
    table_names: list[str] = args.table_names

    for table in table_names:
        sqlite_table_to_parquet(
            sqlite_path=db_filename,
            table_name=table,
            parquet_path=str(output_dir / f"{table}.parquet"),
            context_length=context_length,
        )


if __name__ == "__main__":
    main()
