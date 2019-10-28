Example usage (Scanette)
------------------------
To illustrate a typical workflow we shall use the **Scanette** Example.
This is a simple Supermarket Scanner application, where a shopper can take a
scanner around the store and scan items as they add them to their basket, setting
aside any items that do not scan correctly.
At the checkout, the cashier can manually add any items that would not scan
correctly, and may chose to do a random 'control' on that basket to check if
all items have been scanned.  

We have a CSV log file of the interactions of all the scanners with a server,
covering a short period of time, such as a few hours of shopping.
The typical steps we might want to go through are:

  1. READ the CSV log file into an Agilkia ``TraceSet`` object (one long trace).
  2. SPLIT that one long trace into many smaller traces, based on the customer ID.
  3. CLUSTER those traces, to see typical customer behaviours, and unusual customer behaviours.
  4. VISUALISE those clusters, as a 2D graph, and as sets of short strings.
  5. TRAIN a machine learning model to behave like customers.
  6. GENERATE test traces from those ML models.
  7. EXECUTE those test traces on the server.

Here is example code that goes through the first four of these steps::

    """
    Example analysis of Scanette logs - new CSV format.
    
    Reads Scanette CSV files with these columns:
    # Columns Docs from Frederick, 2019-10-18.
      0: id is an identifier of the line (some numbers may be missing)
      1: timestamp is in Linux format with three extra digits for milliseconds.
      2: sessionID provides the identifier of the session - each client is different.
      3: objectInstance is the object instance on which the operation is invoked.
      4: operation is the name of the operation (action).
      5: parameters is a list of the parameter values, or [] if there are no parameters.
      6: result is the status code returned (? means that the operation does
        not return anything - void)
    
    Created on Thu Oct 17 16:45:39 2019
    
    @author: utting@usc.edu.au
    """
    
    import csv
    from pathlib import Path
    from datetime import datetime, date, time
    from sklearn.cluster import MeanShift
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import matplotlib.cm as pltcm
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    
    import agilkia
    
    # %%
    
    def read_traces_csv(path: Path) -> agilkia.TraceSet:
        # print("now=", datetime.now().timestamp())
        with path.open("r") as input:
            trace1 = agilkia.Trace([])
            for line in csv.reader(input):
                # we ignore the line id.
                timestr = line[1].strip()
                timestamp = date.fromtimestamp(int(timestr) / 1000.0)
                # print(timestr, timestamp.isoformat())
                sessionID = line[2].strip()
                objInstance = line[3].strip()
                action = line[4].strip()
                paramstr = line[5].strip()
                result = line[6].strip()
                # now we identify the main action, inputs, outputs, etc.
                if paramstr == "[]":
                    inputs = {}
                else:
                    if  paramstr.startswith("[") and paramstr.endswith("]"):
                        # strip the brackets off the (single) value.
                        paramstr = paramstr[1:-1]
                    inputs = {"param" : paramstr}
                if result == "?":
                    outputs = {}
                else:
                    outputs = {'Status': float(result)}
                others = {
                        'timestamp': timestamp,
                        'sessionID': sessionID,
                        'object': objInstance
                        }
                event = agilkia.Event(action, inputs, outputs, others)
                trace1.append(event)
        traceset = agilkia.TraceSet([])
        traceset.append(trace1)
        return traceset
    
    
    # %% Read traces and save in the Agilkia JSON format.
    
    traceset = read_traces_csv(Path("127.0.0.1-1571403244552.csv"))
    traceset.set_event_chars({"scanner": ".", "abandon": "a", "supprimer": "s", "ajouter": "+"})
    traceset.save_to_json(Path("log_one.json"))
    
    print(traceset.get_event_chars())  # default summary char for each kind of event.
    
    print(str(traceset[0])[:1000], "...")  # everything is in one big trace initially.
    
    
    # %% Split into separate traces, first based on Scanette number.
    
    print("\n\n==== grouped by sessionID number ====")
    traceset3 = traceset.with_traces_grouped_by("sessionID", property=True)
    for tr in traceset3:
        print("   ", tr)
    
    
    # %% Looks good, so save these split-up traces.
    
    traceset3.save_to_json(Path("log_split.json"))
    
    
    # %% Get some data about each trace, to use for clustering traces.
    
    data = traceset3.get_trace_data(method="action_counts")  # or add: method="action_status_counts"
    print(data.sum().sort_values())
    
    # %% Now cluster the traces (default MeanShift)
    
    num_clusters = traceset3.create_clusters(data)
    print(num_clusters, "clusters found")
    for i in range(num_clusters):
        print(f"Cluster {i}:")
        for tr in traceset3.get_cluster(i):
            print(f"    {tr}")
    
    # %% Visualise clusters (using TSNE)
    
    traceset3.visualize_clusters()
