# Agilkia: AGILe (K)Coverage with Intelligence Artificial

A Python toolkit for building smart testing tools for web services and web sites.


# Key Features:

* Automated testing of SOAP web services with WSDL descriptions.
* Manage sets of traces (load/save to JSON, etc.).
* Convert traces to Pandas DataFrame for data analysis / machine learning.
* Generate random tests, or 'smart' tests from an ML model.
* Split traces into smaller traces (sessions).
* Cluster traces on various criteria, to see common / rare behaviours.
* Visualise clusters of tests.
 

## About the Name

Apart from the clumsy acronym, the name Agilkia was chosen because
it is closely associated with the name 'Philae', and this tool
came out of the Philae research project.

Agilkia is an island in the reservoir of the Aswan Low Dam, 
downstream of the Aswan Dam and Lake Nasser, Egypt.  
It is the current location of the ancient temple of Isis, which was 
moved there from the islands of Philae after dam water levels rose.
    
Agilkia was also the name given to the first landing place of the
Philae landing craft on the comet 67P/Churyumov–Gerasimenko,
during the Rosetta space mission.

This 'agilkia' library is part of the Philae research project:

    http://projects.femto-st.fr/philae/en

It is open source software under the MIT license.
See LICENSE.txt


# Example Usage

Here is a simple example of initial random testing of a web service
running on the URL http://localhost/cash:
```
import agilkia

# sample input values for named parameters
input_values = {
    "username"  : ["TestUser"],
    "password"  : ["<GOOD_PASSWORD>"] * 9 + ["bad-pass"],  # wrong 10% of time
    "version"   : ["2.7"] * 9 + ["2.6"],      # old version 10% of time
    "account"   : ["acc100", "acc103"],       # test these two accounts equally
    "deposit"   : [i*100 for i in range(8)],  # a range of deposit amounts
}

def first_test():
    tester = agilkia.RandomTester("http://localhost/cash",
        parameters=input_values)
    tester.set_username("TestUser")   # will prompt for password
    for i in range(10):
        tr = tester.generate_trace(length=30)
        print(f"========== trace {i}:\n  {tr}")
```

