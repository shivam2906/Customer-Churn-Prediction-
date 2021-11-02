import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from sklearn.ensemble import RandomForestClassifier 
#from sklearn import metrics
from flask import Flask, request, render_template
import pickle


app = Flask(__name__, static_url_path = "/static")  # this peace of code help falsk to render to this folder 


df1 = pd.read_csv("test_telo.csv")

q = "" 

# route directory will open home page   
@app.route("/")
def lodepage():
    return render_template("home.html", query ="")



    
 #'''
 #['SeniorCitizen',
# 'Partner',
 #'Dependents',
 #'MultipleLines',
 #'InternetService',
 #'OnlineSecurity',
 #'OnlineBackup',
 #'DeviceProtection',
 #'TechSupport',
 #'StreamingTV',
 #'StreamingMovies',
 #'PaperlessBilling',
 #'PaymentMethod',
 #'tenure'
# 'MonthlyCharges']  
 
 #'''


@app.route("/predict", methods=["POST"])
def predict():
    # rendering inputs from gui 
    input_1 = request.form['SeniorCitizen']
    input_2 = request.form['Partner']
    input_3 = request.form['Dependents']
    input_4 = request.form['MultipleLines']
    input_5 = request.form['InternetService']
    input_6 = request.form['OnlineSecurity']
    input_7 = request.form['OnlineBackup']
    input_8 = request.form['DeviceProtection']
    input_9 = request.form['TechSupport']
    input_10 = request.form['StreamingTV']
    input_11 = request.form['StreamingMovies']
    input_12 = request.form['PaperlessBilling']
    input_13 = request.form['PaymentMethod']
    input_14 = int(request.form['tenure'])
    input_15 = int(request.form['MonthlyCharges'])

    
    model = pickle.load(open("model.sav", "rb"))
    
    data = [[input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11,
             input_12, input_13, input_14,input_15]]
    
    new_df = pd.DataFrame(data, columns = ['SeniorCitizen','Partner', 'Dependents','MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                                           'PaymentMethod', 'tenure', 'MonthlyCharges'])
    

    df2 = pd.concat([df1, new_df], ignore_index = True)
    
    X = pd.get_dummies(df_2[['SeniorCitizen','Partner', 'Dependents','MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                                           'PaymentMethod']], drop_first = True)
    sc = MinMaxScaler()
    a = sc.fit_transform(df2[['tenure']])
    b = sc.fit_transform(df2[['MonthlyCharges']])
   
    X['tenure'] = a
    X['MonthlyCharges'] = b
    
    single = model.predict(X.tail(1))
    probablity = model.predict_proba(X.tail(1))[:,1]
    
    if single==1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
    
    
    return render_template('home.html', output1=o1, output2=o2)

if __name__ == "__main__":
    app.run(debug = True)



