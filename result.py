#!C:\Users\Sohila\AppData\Local\Programs\Python\Python310\python.exe
from tkinter import W
import wave
import webbrowser
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier


traning_dataset = pd.read_csv("pone.0219385.s001.csv")

# replace all an empty cell with 0
traning_dataset.fillna(0, inplace=True)

# x = independent variables
# y = depenpent variables
# i don't won't the first col in csv which contain Id so i remove it from x
x = traning_dataset.iloc[:, 1:-1]
y = traning_dataset.iloc[:, 19]


# pre processing convert csv into array and removing frist row
x = x.to_numpy()
x = x[1:, :]
y = y.to_numpy()
y = y[1:]
# print(x)
# print(y)


# split data set into traning set and testing set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)


# fitting multiple linear regression into traninng set

#regressor = LinearRegression()
#regressor.fit(x_train, y_train)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# predict MAOA Dosage
y_Predict = rf.predict(x_test)
# print(x_test[0][:])


# print(y_Predict)


# accurecy
acc = r2_score(y_test, y_Predict)
print(acc)

'''
f = open("1.html" , W)
message = 
    <html>
    <head></head>
    <body>
    <form action="">
    <input type="text" class="id" >
    <input type="text" class="wave" >
    <input type="text" class="COLWEEKC" >
    <input type="text" class="TIMEPOINT" >
    <input type="text" class="RCT_ARM" >
    <input type="text" class="SITE_t1" >
    <input type="text" class="GENDER_t1" >
    <input type="text" class="AGE_WAVE" >
    <input type="text" class="TECTOT" >
    <input type="text" class="HINSECURITY" >
    <input type="text" class="HDISTRESS" >
    <input type="text" class="PSSTOT" >
    <input type="text" class="AYMHTOT" >
    <input type="text" class="SDQtot" >
    <input type="text" class="prosoc" >
    <input type="text" class="CRIESTOT" >
    <input type="text" class="NewCYRM12" >
    <input type="text" class="MAOA_Allele1" >
    <input type="text" class="MAOA_Allele2" >
    </form>





    <input type="submit" >
    </body>
    </html>
    
f.write(message )
f.close()
webbrowser.open_new_tab("1.html")



'''

'''
def avoid_null(input_value):
    if (input_value) is None:
        input_value == 0 
    else:
        input_value == input_value
    return  numpy.float16(input_value)
'''



id = input("Enter id: ")
WAVE = input("Enter WAVE: ")
COLWEEKC = input("Enter COLWEEKC: ")
TIMEPOINT = input("Enter TIMEPOINT: ")
RCT_ARM = input("Enter RCT_ARM: ")
SITE_t1 = input("Enter SITE_t1: ")
GENDER_t1 = input("Enter GENDER_t1: ")
AGE_WAVE = input("Enter AGE_WAVE: ")
TECTOT = input("Enter TECTOT: ")
HINSECURITY = input("Enter HINSECURITY: ")
HDISTRESS = input("Enter HDISTRESS: ")
PSSTOT = input("Enter PSSTOT: ")
AYMHTOT = input("Enter AYMHTOT: ")
SDQtot = input("Enter SDQtot: ")
prosoc = input("Enter prosoc: ")
CRIESTOT = input("Enter CRIESTOT: ")
NewCYRM12 = input("Enter NewCYRM12: ")
MAOA_Allele1 = input("Enter MAOA_Allele1: ")
MAOA_Allele2 = input("Enter MAOA_Allele2: ")





#form = cgi.FieldStorage()
#id = numpy.float32(form.getvalue("id"))
#WAVE = numpy.float32(form.getvalue("wave"))
#COLWEEKC = numpy.float32(form.getvalue("COLWEEKC"))
#TIMEPOINT = numpy.float32(form.getvalue("TIMEPOINT"))
#RCT_ARM = numpy.float32(form.getvalue("RCT_ARM"))
#SITE_t1 = numpy.float32(form.getvalue("SITE_t1"))
#GENDER_t1 = numpy.float32(form.getvalue("GENDER_t1"))
#AGE_WAVE = numpy.float32(form.getvalue("AGE_WAVE"))
#TECTOT = numpy.float32(form.getvalue("TECTOT"))
#HINSECURITY = numpy.float32(form.getvalue("HINSECURITY"))
#HDISTRESS = numpy.float32(form.getvalue("HDISTRESS"))
#PSSTOT = numpy.float32(form.getvalue("PSSTOT"))
#AYMHTOT = numpy.float32(form.getvalue("AYMHTOT"))
#SDQtot = numpy.float32(form.getvalue("SDQtot"))
#prosoc = numpy.float32(form.getvalue("prosoc"))
#CRIESTOT = numpy.float32(form.getvalue("CRIESTOT"))
#NewCYRM12 = numpy.float32(form.getvalue("NewCYRM12"))
#MAOA_Allele1 = numpy.float32(form.getvalue("MAOA_Allele1"))
#MAOA_Allele2 = numpy.float32(form.getvalue("MAOA_Allele2"))

input_data = [WAVE, COLWEEKC, TIMEPOINT, RCT_ARM, SITE_t1, GENDER_t1, AGE_WAVE, TECTOT, HINSECURITY, HDISTRESS, PSSTOT,AYMHTOT, SDQtot, prosoc, CRIESTOT, NewCYRM12, MAOA_Allele1, MAOA_Allele2]

new_input_data = numpy.reshape(input_data, (1, -1))
# new_input_data=new_input_data[0][1:].astype(float)
# print(new_input_data)

res = rf.predict(new_input_data)
print(res)


def show_result(result):
    f = open("result.html", W)
    message = '''
    <html>
    <head>
    <link rel="stylesheet" href="css/result.css">
    </head>
    <body>
    <div class="card">
    <h3>your result</h3>
    <p>{result} </p> 

    </div>
    </body>
    </html>
    '''.format(result=result)
    f.write(message)
    f.close()
    webbrowser.open_new_tab("result.html")


if res == '0':
    show_result('low activity')
    print('low activity')
elif res == '0.5':
    show_result('hetrozygos activity')
    print('hetrozygos activity')
else:
    show_result('high activity')
    print('high activity')
print(res)
