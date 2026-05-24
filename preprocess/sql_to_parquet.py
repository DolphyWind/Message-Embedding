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
    no_user_tokens: bool = False,
) -> None:
    print(f"  Loading table '{table_name}' from {sqlite_path} ...")
    conn = sqlite3.connect(sqlite_path)
    try:
        query = (
            f"SELECT * FROM {table_name} "
            f"ORDER BY DATETIME(timestamp) ASC "
        )
        df: pd.DataFrame = pd.read_sql_query(query, conn)
        print(f"  Loaded {len(df)} rows from '{table_name}'.")
        if df.empty:
            print(f"  Table '{table_name}' is empty, skipping.")
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
            if no_user_tokens:
                positive: str = ' '.join(group)
            else:
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

        print(f"  Writing {len(new_df)} windows to {parquet_path} ...")
        new_df.to_parquet(
            parquet_path,
            engine="fastparquet",
            index=False,
        )
        print(f"  Done with '{table_name}'.")
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
    parser.add_argument(
        "--no_user_tokens",
        action="store_true",
        help="Do not use any user tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_filename: str = args.input_file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_length: int = args.context_len
    table_names: list[str] = args.table_names or get_all_table_names(db_filename)
    stride: int = args.stride
    no_user_tokens: bool = args.no_user_tokens

    print(f"Processing {len(table_names)} table(s)")
    for table in table_names:
        sqlite_table_to_parquet(
            sqlite_path=db_filename,
            table_name=table,
            parquet_path=str(output_dir / f"{table}.parquet"),
            context_length=context_length,
            stride=stride,
            no_user_tokens=no_user_tokens,
        )


def get_all_table_names(sqlite_path: str) -> list[str]:
    print(f"Querying all table names from {sqlite_path} ...")
    conn = sqlite3.connect(sqlite_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Found {len(tables)} tables")
        return tables
    finally:
        conn.close()


if __name__ == "__main__":
    main()
