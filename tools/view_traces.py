# -*- coding: utf-8 -*-
"""
View the traces within an Agilkia TraceSet.

It prints a one-line summary of each trace, plus some general statistics.
If the TraceSet is clustered, traces will be displayed in clusters, by default.

TODO: allow user to specify the 'Ok' status value (e.g. --ok=200 for HTTP results)

@author: m.utting@uq.edu.au
"""

import pandas as pd
import argparse
from pathlib import Path
import textwrap

import agilkia

INDENT = "   "  # prefix for each trace (plus one extra space)


def make_action_status_table(df: pd.DataFrame, ok=0) -> pd.DataFrame:
    """From TraceSet DataFrame, creates a table of Actions showing how many got Ok vs Error."""
    good = df[df.Status == ok].groupby("Action").size()
    err = df[df.Status != ok].groupby("Action").size()
    data = pd.DataFrame({"Ok": good, "Err": err})
    data.fillna(0, inplace=True, downcast="infer")
    data["Total"] = data.Ok + data.Err
    totals = pd.DataFrame([data.sum().rename("Total")])
    # add Totals row at bottom
    return pd.concat([data, totals])


def main():
    """A command line program that prints a set of traces, plus some summary statistics."""
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", help="show more information, such as the event-to-char map.",
        action="store_true")
    parser.add_argument("-e", "--eventchars", help="a csv file containing an event-to-char map.")
    parser.add_argument("-n", "--noclusters", help="view traces in-order, without clusters.",
        action="store_true")
    parser.add_argument("-r", "--repeats", help="compress repeated events with action=REPEATS")
    parser.add_argument("-s", "--status", help="color events with non-zero status red, to highlight errors",
        action="store_true")
    parser.add_argument("--ok", help="specify the status value that represents success (default 0).", 
        default="0", metavar="NUM", type=int)
    parser.add_argument("traceset", help="an Agilkia traceset file (*.json)")
    args = parser.parse_args()
    # print(f"Args are:", args)
    traceset = agilkia.TraceSet.load_from_json(Path(args.traceset))
    actions = agilkia.all_action_names(traceset.traces)
    if args.eventchars:
        mapfile = pd.read_csv(args.eventchars, header=None)
        # we assume this has just two columns: 0=action_name and 1=char.
        char_map = dict(zip(mapfile.iloc[:, 0], mapfile.iloc[:, 1]))
        # print("given map=", char_map)
        traceset.set_event_chars(char_map)
    # print("final map=", char_map)
    repeats = [] if args.repeats is None else [args.repeats]
    if traceset.is_clustered() and not args.noclusters:
        clusters = traceset.get_clusters()
        for c in range(traceset.get_num_clusters()):
            print(f"Cluster {c}:")
            for (i, tr) in enumerate(traceset.traces):
                if clusters[i] == c:
                    print(INDENT, tr.to_string(compress=repeats, color_status=args.status))
    else:
        for tr in traceset.traces:
            print(INDENT, tr.to_string(compress=repeats, color_status=args.status))
    if args.verbose:
        print("==== event chars ====")
        ev_chars = traceset.get_event_chars()
        for action,ch in ev_chars.items():
            print(f"    {action:20s}    {ch}")
    print("==== statistics ====")
    df = traceset.to_pandas()
    statuses = df.Status.value_counts()
    percent_ok = 100.0 * statuses.get(args.ok, 0) / df.shape[0]
    # print(df.head())
    print(f"Number of traces     : {len(traceset.traces)}")
    print(f"Average trace length : {df.groupby('Trace').count().Action.mean():.2f}")
    print(f"Number of clusters   : {traceset.get_num_clusters()}")
    print(f"Number of events     : {df.shape[0]}")
    print(f"Number of event kinds: {len(actions)}")
    print(textwrap.indent(str(make_action_status_table(df, ok=args.ok)), "    "))
    print(f"Percent of status=ok : {percent_ok:.2f}%")
    error_counts = df.groupby("Error").Action.count()
    if len(error_counts) > 1:
        print(f"Categories of errors : ({100.0 - percent_ok:.2f}% total)")
        print(textwrap.indent(str(error_counts), "    "))


if __name__ == "__main__":
    main()
