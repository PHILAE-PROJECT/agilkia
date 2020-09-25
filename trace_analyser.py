# -*- coding: utf-8 -*-
"""
Analyse results of a testing run.

It prints a one-line summary of each trace, plus some general statistics.
Optionally, it can also report how many rows were added to each database table.

@author: m.utting@uq.edu.au
"""

import pandas as pd
import argparse
from pathlib import Path
import textwrap

import agilkia


def read_database_changes(before_csv: str, after_csv: str) -> pd.DataFrame:
    """Reads two files of database row counts and calculates tuples added to each table.

    Args:
        before_csv: name of CSV file containing the 'before' counts.
        after_csv: name of CSV file containing the 'after' counts.

    Returns:
        A Pandas table with an 'added' column for how many rows were added to each table.
    """
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


def make_action_status_table(df: pd.DataFrame) -> pd.DataFrame:
    """From TraceSet DataFrame, creates a table of Actions showing how many got Ok vs Error."""
    ok = df[df.Status == 0].groupby("Action").size()
    err = df[df.Status != 0].groupby("Action").size()
    data = pd.DataFrame({"Ok": ok, "Err": err})
    data.fillna(0, inplace=True, downcast="infer")
    data["Total"] = data.Ok + data.Err
    totals = data.sum().rename("Total")
    # add Totals row at bottom
    data = data.append(totals)
    # total = df.shape[0]  # number of rows = total event count
    # percents = (totals * 100.0 / total).rename("Percent")
    # data = data.append(percents)
    return data


def main():
    """A command line program that gives an overview of a set of generated traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-a", "--after", help="database row counts AFTER testing (*.csv)")
    parser.add_argument("-b", "--before", help="database row counts BEFORE testing (*.csv)")
    parser.add_argument("-c", "--chars", help="name of action-to-CHAR mapping file (*.csv)")
    parser.add_argument("-r", "--repeats", help="remove REPEATS of this action")
    parser.add_argument("-s", "--status", help="show STATUS in color (red=error)",
                        action="store_true")
    parser.add_argument("traces", help="traces file (*.json)")
    args = parser.parse_args()
    # print(f"Args are:", args)
    if args.before and args.after:
        changes = read_database_changes(args.before, args.after)
        nonzero = changes[changes.added > 0].sort_values(by="added", ascending=False)
        print("==== database changes ====")
        print(nonzero)
    traceset = agilkia.TraceSet.load_from_json(Path(args.traces))
    actions = agilkia.all_action_names(traceset.traces)
    if args.chars:
        mapfile = pd.read_csv(args.chars, header=None)
        # we assume this has just two columns: 0=action_name and 1=char.
        char_map = dict(zip(mapfile.iloc[:, 0], mapfile.iloc[:, 1]))
        # print("given map=", char_map)
        traceset.set_event_chars(char_map)
    # print("final map=", char_map)
    repeats = [] if args.repeats is None else [args.repeats]
    for tr in traceset.traces:
        print(tr.to_string(compress=repeats, color_status=args.status))
    print("==== statistics ====")
    df = traceset.to_pandas()
    # print(df.head())
    print(f"Number of traces     : {len(traceset.traces)}")
    print(f"Average trace length : {df.groupby('Trace').count().Action.mean()}")
    print(f"Number of events     : {df.shape[0]}")
    print(f"Number of event kinds: {len(actions)}")
    print(textwrap.indent(str(make_action_status_table(df)), "    "))
    statuses = df.Status.value_counts()
    percent_ok = 100.0 * statuses[0] / df.shape[0]
    print(f"Detailed error counts: ({100.0 - percent_ok:.2f}%)")
    print(textwrap.indent(str(df.groupby("Error").count().Action), "    "))
    print(f"Percent of status ok : {percent_ok:.2f}%")


if __name__ == "__main__":
    main()
