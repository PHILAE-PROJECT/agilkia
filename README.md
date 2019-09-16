# Agilkia: AGILe (K)Coverage with Intelligence Artificial

Automated smart testing tools for web services.

This 'agilkia' library is part of the Philae research project.

It currently supports testing SOAP web services with WSDL descriptions.

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



# Example Usage

Here is a simple example of initial dumb random testing of a web service
called "CASH" running on a server http://localhost/:
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
    tester = agilkia.RandomTester("http://localhost",
        services=["CASH"],
        parameters=input_values)
    tester.set_username("TestUser")   # will prompt for password
    for i in range(10):
        tr = tester.generate_trace(length=30)
        print(f"========== trace {i}:\n  {tr}")
```
 
