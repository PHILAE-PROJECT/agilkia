# -*- coding: utf-8 -*-
"""
Analyse results of a testing run.

It prints a one-line summary of each trace, plus some general statistics.
Optionally, it can also report how many rows were added to each database table.

@author: utting@usc.edu.au
"""

import pandas as pd
import argparse
import textwrap

import agilkia


def read_database_changes(before_csv: str, after_csv: str) -> pd.DataFrame:
    before = pd.read_csv(before_csv)
    after = pd.read_csv(after_csv)
    col_msg = "ERROR: {} must have columns 'name', 'row_count', ..."
    if list(before.columns[0:2]) != ["name", "row_count"]:
        print(col_msg.format(before_csv))
    if list(after.columns[0:2]) != ["name", "row_count"]:
        print(col_msg.format(before_csv))
    # we use inner join to get the intersection of the two sets of tables.
    changes = pd.merge(before, after, how="inner", on="name", suffixes=("_before", "_after"))
    changes["added"] = changes["row_count_after"] - changes["row_count_before"]
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-b", "--before", help="database row counts before testing (*.csv)")
    parser.add_argument("-a", "--after", help="database row counts after testing (*.csv)")
    parser.add_argument("-c", "--compress", help="compress repeats of this action")
    parser.add_argument("-m", "--map", help="name of action-to-char mapping file (*.csv)")
    parser.add_argument("-s", "--status", help="show statusin color (red=error)",
                        action="store_true")
    parser.add_argument("traces", help="traces file (*.json)")
    args = parser.parse_args()
    # print(f"Args are:", args)
    if args.before and args.after:
        changes = read_database_changes(args.before, args.after)
        nonzero = changes[changes.added > 0].sort_values(by="added", ascending=False)
        print("==== database changes ====")
        print(nonzero)
    traces = agilkia.load_traces_from_json(args.traces)
    actions = agilkia.all_action_names(traces)
    if args.map:
        mapfile = pd.read_csv(args.map, header=None)
        # we assume this has just two columns: 0=action_name and 1=char.
        char_map = dict(zip(mapfile.iloc[:, 0], mapfile.iloc[:, 1]))
        # print("given map=", char_map)
    else:
        char_map = {}
    char_map = agilkia.default_map_to_chars(actions, given=char_map)
    # print("final map=", char_map)
    compress = [] if args.compress is None else [args.compress]
    for tr in traces:
        print(agilkia.trace_to_string(tr, char_map, compress=compress, color_status=args.status))
    print("==== statistics ====")
    df = agilkia.traces_to_pandas(traces)
    # print(df.head())
    print(f"Number of traces    : {len(traces)}")
    print(f"Average trace length: {df.groupby('trace').count().action.mean()}")
    statuses = df.Status.value_counts()
    percent_ok = 100.0 * statuses[0] / df.shape[0]
    print(f"Percent of status ok: {percent_ok:.2f}%")
    print(f"Detailed error frequencies:")
    print(textwrap.indent(str(df.groupby("Error").count().action), "    "))


if __name__ == "__main__":
    main()
