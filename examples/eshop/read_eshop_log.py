# -*- coding: utf-8 -*-
"""
Example project to read e-Shop logs and save in Agilkia JSON format.

It can also split the traces into sessions based on the SessionID.

It uses the Philae 'Agilkia' library, which can be installed by::

    conda install -c mark.utting agilkia

Created April 2021

@author: m.utting@uq.edu.au
"""

import json
from pathlib import Path
from datetime import datetime
import sys

import agilkia

# %%

def read_traces_log(path: Path) -> agilkia.TraceSet:
    """Reads the given log file as a single long trace.

    Note that the sequence id number and datetime stamp are recorded as meta data of each event,
    with the datatime stamp converted to a Python datetime object.

    The "sessionID" and date-time is recorded as meta-data of the event.
    The "function" is used as the action name.
    The "customerID" and "controller" are treated as inputs.
    The "httpResponseCode" is the output and is renamed to 'Status'.

    For example, theis line of an input file::

      2021-03-04 10:19:58 - {"sessionID":"0fb311c015e861eebebd596d90","customerID":0,"controller":"ControllerCommonHome",
        "function":"index","data":{"route":"common\/home"},"httpResponseCode":200}
    
    will translate into this Event::

        Event("index", inputs1, outputs1, meta1)
        where
          inputs1 = {"customerID": "0", "controller": "ControllerCommonHome"}
          outputs1 = {"Status": 200}
          meta1 = {"sessionID": 0fb311c015e861eebebd596d90, "timestamp": <2021-03-04T10:19:58>}
    """
    # print("now=", datetime.now().timestamp())
    with path.open("r") as input:
        trace1 = agilkia.Trace([])
        for line in input:
            sep = line.find(" - ")
            if sep <= 0:
                print("WARNING: skipping line:", line)
                continue

            timestamp = datetime.strptime(line[0:sep], '%Y-%m-%d %H:%M:%S')
            contents = json.loads(line[sep+3:])
            # print(timestamp, contents)

            sessionID = contents["sessionID"]
            customerID = contents["customerID"]
            controller = contents["controller"]
            function = contents["function"]
            data = contents["data"]
            route = data["route"]
            del data["route"]  # route is the main action, so does not need to be an input too
            response = int(contents["httpResponseCode"])
            # now we identify inputs, outputs, and meta-data.
            inputs = {
                    'customerID': customerID,
                    'sessionID': sessionID,
                    }
            inputs.update(data)  # add all the data keys and values.
            outputs = {
                'httpResponseCode': response,
                'Status': 0 if 200 <= response < 300 else 1}
            meta = {
                    'timestamp': timestamp,
                    'controller': controller,
                    }
            event = agilkia.Event(route, inputs, outputs, meta)
            trace1.append(event)
            # See which data keys are associated with each type of event?
            # print(function, route, sorted(list(data.keys())))
    traceset = agilkia.TraceSet([])
    traceset.append(trace1)
    return traceset


# %% How to visualise traces

# We use lowercase or punctuation for common functions, uppercase for less common ones.
abbrev_chars = {
    "add": "+",
    "addAddress": "A",
    "addCustomer": "C",
    "addOrderHistory": "O",
    "addReview": "R",
    "agree": "Y",  # like "Yes"
    "alert": "!",
    "confirm": "y",  # for "yes"
    "country": "c",
    "coupon": "#",
    "currency": "$",
    "customfield": "f",
    "delete": "d",
    "deleteAddress": "D",
    "edit": "e",
    "editPassword": "E",
    "getRecurringDescription": "X",
    "index": ".",
    "login": "L",
    "quote": "q",
    "remove": "x",
    "reorder": "&",
    "review": "r",
    "save": "s",
    "send": "S",
    "shipping": "^",
    "success": "=",
    "voucher": "v",
    "write": "w"
    }


# Since there are so many different types of actions, we use two chars for each one.
# First char is:
#   '.' for product/
#   '^' for checkout
#   '@' for account
#   '+' for extension
#   '?' for information
#   '=' for common
#   '#' for tool
abbrev_chars = {
    "account/account": "@@",              #      29 account
    "account/account/country": "@c",      #       1 account
    "account/address": "@a",              #       6 account
    "account/address/add": "@A",          #       1 account
    "account/address/delete": "@-",       #       2 account
    "account/download": "@d",             #       2 account
    "account/edit": "@e",                 #       2 account
    "account/login": "@L",                #      63 account
    "account/logout": "@l",               #      17 account
    "account/newsletter": "@n",           #       4 account
    "account/order": "@o",                #      13 account
    "account/order/reorder": "@O",        #       2 account
    "account/password": "@p",             #       7 account
    "account/recurring": "@2",            #       2 account
    "account/register": "@r",             #      38 account
    "account/register/customfield": "@f", #      24 account
    "account/return/add": "@+",           #       3 account
    "account/reward": "@R",               #       2 account
    "account/success": "@s",              #       3 account
    "account/transaction": "@t",          #       3 account
    "account/voucher": "@v",              #       5 account
    "account/wishlist": "@w",             #      25 account
    "account/wishlist/add": "@W",         #      35 account
    "checkout/cart": "^_",                #     141 checkout
    "checkout/cart/add": "^+",            #     186 checkout
    "checkout/cart/edit": "^e",           #      21 checkout
    "checkout/cart/remove": "^-",         #      32 checkout
    "checkout/checkout": "^^",            #      90 checkout
    "checkout/checkout/country": "^c",    #     151 checkout
    "checkout/checkout/customfield": "^f", #      50 checkout
    "checkout/confirm": "^C",             #      27 checkout
    "checkout/guest": "^g",               #      33 checkout
    "checkout/guest/save": "^s",          #      29 checkout
    "checkout/guest_shipping": "^G",      #      16 checkout
    "checkout/login": "^L",               #      53 checkout
    "checkout/login/save": "^S",          #      10 checkout
    "checkout/payment_address": "^a",     #      38 checkout
    "checkout/payment_address/save": "^A", #      13 checkout
    "checkout/payment_method": "^p",      #      29 checkout
    "checkout/payment_method/save": "^P", #      43 checkout
    "checkout/register": "^r",            #      17 checkout
    "checkout/register/save": "^R",       #      29 checkout
    "checkout/shipping_address": "^d",    #      25 checkout
    "checkout/shipping_address/save": "^D", #      14 checkout
    "checkout/shipping_method": "^m",     #      28 checkout
    "checkout/shipping_method/save": "^M", #      26 checkout
    "checkout/success": "^y",             #      26 checkout
    "common/currency/currency": "=c",     #      15 common
    "common/home": "=h",                  #     122 common
    "extension/payment/cod/confirm": "+p",                #     126 extension
    "extension/total/coupon/coupon": "+c",                #      27 extension
    "extension/total/shipping/country": "+C",             #     122 extension
    "extension/total/shipping/quote": "+q",               #      19 extension
    "extension/total/shipping/shipping": "+s",            #      21 extension
    "extension/total/voucher/voucher": "+v",              #      13 extension
    "information/contact": "?c",                  #      13 information
    "information/contact/success": "?C",          #       2 information
    "information/information": "?i",              #      11 information
    "information/information/agree": "?I",        #       2 information
    "information/sitemap": "?s",          #       3 information
    "product/category": ".c",             #     321 product
    "product/compare": ".C",              #      31 product
    "product/compare/add": ".a",          #      30 product
    "product/manufacturer": ".m",         #       4 product
    "product/product": ".p",              #     165 product
    "product/product/getRecurringDescription": ".d", #      19 product
    "product/product/review": ".r",               #     154 product
    "product/product/write": ".w",                #      38 product
    "product/search": ".s",               #      36 product
    "product/special": ".S",              #       1 product
    "tool/upload": "#u",                  #       6 tool
}

# check that the abbrev chars are all unique.
dups = [v for v in abbrev_chars.values() if list(abbrev_chars.values()).count(v) > 1]
assert len(set(abbrev_chars.keys())) == len(set(abbrev_chars.values())), "duplicates: " + str(dups)

# %% Read traces and save in the Agilkia JSON format.

def read_split_save(name: str, split: bool, verbose: bool = False):
        path = Path(name)
        traces = read_traces_log(path)
        traces.set_event_chars(abbrev_chars)
        msg = ""
        if split:
            traces = traces.with_traces_grouped_by(key=(lambda ev: ev.inputs["sessionID"]))
            path2 = path.with_suffix(".split.json")
        else:
            path2 = path.with_suffix(".json")
        if verbose:
            print("# abbrev_chars =", abbrev_chars)
            for tr in traces:
                print(tr)
        print(f"  {path} -> {path2} [{len(traces)} traces{msg}]")
        traces.save_to_json(path2)

# %%

def main(args):
    start = 1
    split = False
    verbose = False
    if start < len(args) and args[start] == "--split":
        start += 1
        split = True
    if start < len(args) and args[start] == "--verbose":
        start += 1
        verbose = True
    files = args[start:]
    if len(files) > 0:
        for name in files:
            read_split_save(name, split=split, verbose=verbose)
    else:
        script = args[0] or "read_eshop_log.py"
        print("This script converts e-shop log files into Agilkia *.json trace files.")
        print("If the --split argument is given, it will also split traces by session IDs.")
        print("If the --verbose argument is given, it will print the trace(s).")
        print(f"Setup: conda install -c mark.utting agilkia")
        print(f"Usage: python {script} [--split] [--verbose] log1.txt log2.txt ...")


# %%

if __name__ == "__main__":
    main(sys.argv)

