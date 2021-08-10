from flask import Flask, render_template, request
import jsonify
import requests
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model1 = joblib.load(open('svclassifier.pkl', 'rb'))
model2 = joblib.load(open('xgb_model_df1.pkl', 'rb'))
df_port = pd.read_csv('student-por-sep.csv')
df_mat = pd.read_csv('student-mat-sep.csv')
df = pd.concat([df_mat,df_port],axis = 0)
df.reset_index(drop=True, inplace=True)
scaler = StandardScaler()
var = ['age','Medu','Fedu','traveltime','studytime','failures','freetime','famrel','goout','Dalc','Walc','health',
       'G1','G2','G3']
df[var] = scaler.fit_transform(df[var])

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

 
standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        Medu =int(request.form['Medu'])
        Fedu =int(request.form['Fedu'])
        traveltime =int(request.form['traveltime'])
        studytime =int(request.form['studytime'])
        failures =int(request.form['failures'])
        famrel =int(request.form['famrel'])
        freetime =int(request.form['freetime'])
        goout =int(request.form['goout'])
        Dalc =int(request.form['Dalc'])
        Walc =int(request.form['Walc'])
        health =int(request.form['health'])
        G1 =int(request.form['G1'])        
        G2 =int(request.form['G2'])
        G3 =int(request.form['G3'])
        school_MS =request.form['school_MS']
        if(school_MS=='MS'):
            school_MS =1
        else:
            school_MS =0
            
        sex_M=request.form['sex_M']
        if(sex_M=='M'):
            sex_M=1
        else:
            sex_M=0	
            
        address_U=request.form['address_U']
        if(address_U=='U'):
            address_U=1
        else:
            address_U=0
            
        famsize_LE3 =request.form['famsize_LE3']
        if(famsize_LE3=='LE3'):
            famsize_LE3=1
        else:
            famsize_LE3=0
            
        Pstatus_T=request.form['Pstatus_T']
        if(Pstatus_T=='T'):
            Pstatus_T=1
        else:
            Pstatus_T=0

        Mjob=request.form['Mjob']
        if(Mjob =='health'):
            Mjob_health = 1
            Mjob_other  = 0
            Mjob_services = 0
            Mjob_teacher = 0           
        elif(Mjob == 'other'):
            Mjob_health = 0
            Mjob_other  = 1
            Mjob_services = 0
            Mjob_teacher = 0 
        elif(Mjob == 'services'):
            Mjob_health = 0
            Mjob_other  = 0
            Mjob_services = 1
            Mjob_teacher = 0 
        elif(Mjob == 'teacher'):
            Mjob_health = 0
            Mjob_other  = 0
            Mjob_services = 0
            Mjob_teacher = 1 
        else:
            Mjob_health = 0
            Mjob_other  = 0
            Mjob_services = 0
            Mjob_teacher = 0 
        
        Fjob=request.form['Fjob']
        if(Fjob =='health'):
            Fjob_health = 1
            Fjob_other  = 0
            Fjob_services = 0
            Fjob_teacher = 0           
        elif(Fjob == 'other'):
            Fjob_health = 0
            Fjob_other  = 1
            Fjob_services = 0
            Fjob_teacher = 0 
        elif(Fjob == 'services'):
            Fjob_health = 0
            Fjob_other  = 0
            Fjob_services = 1
            Fjob_teacher = 0 
        elif(Fjob == 'teacher'):
            Fjob_health = 0
            Fjob_other  = 0
            Fjob_services = 0
            Fjob_teacher = 1 
        else:
            Fjob_health = 0
            Fjob_other  = 0
            Fjob_services = 0
            Fjob_teacher = 0   
            
        reason=request.form['reason']
        if(reason =='home'):
            reason_home = 1
            reason_other  = 0
            reason_reputation = 0        
        elif(reason == 'other'):
            reason_home = 0
            reason_other  = 1
            reason_reputation = 0  
        elif(reason == 'reputation'):
            reason_home = 0
            reason_other  = 0
            reason_reputation = 1 
        else:
            reason_home = 0
            reason_other  = 0
            reason_reputation = 0  
            
        guardian=request.form['guardian']
        if(guardian =='mother'):
            guardian_mother = 1
            guardian_other  = 0 
        elif(guardian == 'other'):
            guardian_mother = 0
            guardian_other  = 1
        else:
            guardian_mother = 0
            guardian_other  = 0
                
        schoolsup_yes=request.form['schoolsup_yes']
        if(schoolsup_yes=='yes'):
            schoolsup_yes=1
        else:
            schoolsup_yes=0

        famsup_yes=request.form['famsup_yes']
        if(famsup_yes=='yes'):
            famsup_yes=1
        else:
            famsup_yes=0
            
        paid_yes=request.form['paid_yes']
        if(paid_yes=='yes'):
            paid_yes=1
        else:
            paid_yes=0

        activities_yes=request.form['activities_yes']
        if(activities_yes=='yes'):
            activities_yes=1
        else:
            activities_yes=0
            
        nursery_yes=request.form['nursery_yes']
        if(nursery_yes=='yes'):
            nursery_yes=1
        else:
            nursery_yes=0

        higher_yes=request.form['higher_yes']
        if(higher_yes=='yes'):
            higher_yes=1
        else:
            higher_yes=0

        internet_yes=request.form['internet_yes']
        if(internet_yes=='yes'):
            internet_yes=1
        else:
            internet_yes=0

        romantic_yes=request.form['romantic_yes']
        if(romantic_yes=='yes'):
            romantic_yes=1
        else:
            romantic_yes=0       
        
        if(G3 <= 8):
            prediction=model_df1.predict([[age,Medu,Fedu,traveltime,studytime,failures,famrel,freetime,goout,Dalc,Walc,health,G1,G2,G3,school_MS,sex_M,address_U,famsize_LE3,Pstatus_T,Mjob_health,Mjob_other,Mjob_services,Mjob_teacher,Fjob_health,Fjob_other,Fjob_services,Fjob_teacher,reason_home,reason_other,reason_reputation,guardian_mother,guardian_other,schoolsup_yes,famsup_yes,paid_yes,activities_yes,nursery_yes,higher_yes,internet_yes,romantic_yes]])
            output=round(prediction[0])
        elif(G3 > 8 and G3 <= 12.5):
            prediction=model_df2.predict([[age,Medu,Fedu,traveltime,studytime,failures,famrel,freetime,goout,Dalc,Walc,health,G1,G2,G3,school_MS,sex_M,address_U,famsize_LE3,Pstatus_T,Mjob_health,Mjob_other,Mjob_services,Mjob_teacher,Fjob_health,Fjob_other,Fjob_services,Fjob_teacher,reason_home,reason_other,reason_reputation,guardian_mother,guardian_other,schoolsup_yes,famsup_yes,paid_yes,activities_yes,nursery_yes,higher_yes,internet_yes,romantic_yes]])
            output=round(prediction[0])
        else:
            prediction=model_df3.predict([[age,Medu,Fedu,traveltime,studytime,failures,famrel,freetime,goout,Dalc,Walc,health,G1,G2,G3,school_MS,sex_M,address_U,famsize_LE3,Pstatus_T,Mjob_health,Mjob_other,Mjob_services,Mjob_teacher,Fjob_health,Fjob_other,Fjob_services,Fjob_teacher,reason_home,reason_other,reason_reputation,guardian_mother,guardian_other,schoolsup_yes,famsup_yes,paid_yes,activities_yes,nursery_yes,higher_yes,internet_yes,romantic_yes]])
            output=round(prediction[0])
        
        if output<0:
            return render_template('index.html',prediction_texts="Sorry")
        else:
            return render_template('index.html',prediction_text="Absences = {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)