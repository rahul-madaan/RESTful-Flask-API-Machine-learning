from flask import Flask, render_template, request
# import jsonify
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
df = pd.concat([df_mat, df_port], axis=0)
df.reset_index(drop=True, inplace=True)
scaler = StandardScaler()
var = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'freetime', 'famrel', 'goout', 'Dalc', 'Walc',
       'health',
       'G1', 'G2', 'G3']
df[var] = scaler.fit_transform(df[var])


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        Medu = int(request.form['Medu'])
        Fedu = int(request.form['Fedu'])
        traveltime = int(request.form['traveltime'])
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        famrel = int(request.form['famrel'])
        freetime = int(request.form['freetime'])
        goout = int(request.form['goout'])
        Dalc = int(request.form['Dalc'])
        Walc = int(request.form['Walc'])
        health = int(request.form['health'])
        G1 = int(request.form['G1'])
        G2 = int(request.form['G2'])
        G3 = int(request.form['G3'])

        school_MS = request.form['school_MS']
        sex_M = request.form['sex_M']
        address_U = request.form['address_U']
        famsize_LE3 = request.form['famsize_LE3']
        Pstatus_T = request.form['Pstatus_T']
        Mjob_health = 0
        Mjob_other = 0
        Mjob_services = 0
        Mjob_teacher = 0
        Mjob = request.form['Mjob']
        if (Mjob == 'health'):
            Mjob_health = 1

        elif (Mjob == 'other'):
            Mjob_other = 1

        elif (Mjob == 'services'):
            Mjob_services = 1

        elif (Mjob == 'teacher'):
            Mjob_teacher = 1

        Fjob_health = 0
        Fjob_other = 0
        Fjob_services = 0
        Fjob_teacher = 0

        Fjob = request.form['Fjob']
        if (Fjob == 'health'):
            Fjob_health = 1

        elif (Fjob == 'other'):
            Fjob_other = 1

        elif (Fjob == 'services'):
            Fjob_services = 1

        elif (Fjob == 'teacher'):
            Fjob_teacher = 1

        reason_home = 0
        reason_other = 0
        reason_reputation = 0

        reason = request.form['reason']
        if (reason == 'home'):
            reason_home = 1

        elif (reason == 'other'):
            reason_other = 1

        elif (reason == 'reputation'):
            reason_reputation = 1

        guardian = request.form['guardian']
        guardian_mother = 0
        guardian_other = 0
        if (guardian == 'mother'):
            guardian_mother = 1
        elif (guardian == 'other'):
            guardian_other = 1

        schoolsup_yes = int(request.form['schoolsup_yes'])

        famsup_yes = int(request.form['famsup_yes'])

        paid_yes = int(request.form['paid_yes'])


        activities_yes = int(request.form['activities_yes'])

        nursery_yes = int(request.form['nursery_yes'])

        higher_yes = int(request.form['higher_yes'])

        internet_yes = int(request.form['internet_yes'])

        romantic_yes = int(request.form['romantic_yes'])



        if (G3 <= 8):
            prediction = model_df1.predict([[age, Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout,
                                             Dalc, Walc, health, G1, G2, G3, school_MS, sex_M, address_U, famsize_LE3,
                                             Pstatus_T, Mjob_health, Mjob_other, Mjob_services, Mjob_teacher,
                                             Fjob_health, Fjob_other, Fjob_services, Fjob_teacher, reason_home,
                                             reason_other, reason_reputation, guardian_mother, guardian_other,
                                             schoolsup_yes, famsup_yes, paid_yes, activities_yes, nursery_yes,
                                             higher_yes, internet_yes, romantic_yes]])
            output = round(prediction[0])
        elif (G3 > 8 and G3 <= 12.5):
            prediction = model_df2.predict([[age, Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout,
                                             Dalc, Walc, health, G1, G2, G3, school_MS, sex_M, address_U, famsize_LE3,
                                             Pstatus_T, Mjob_health, Mjob_other, Mjob_services, Mjob_teacher,
                                             Fjob_health, Fjob_other, Fjob_services, Fjob_teacher, reason_home,
                                             reason_other, reason_reputation, guardian_mother, guardian_other,
                                             schoolsup_yes, famsup_yes, paid_yes, activities_yes, nursery_yes,
                                             higher_yes, internet_yes, romantic_yes]])
            output = round(prediction[0])
        else:
            prediction = model_df3.predict([[age, Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout,
                                             Dalc, Walc, health, G1, G2, G3, school_MS, sex_M, address_U, famsize_LE3,
                                             Pstatus_T, Mjob_health, Mjob_other, Mjob_services, Mjob_teacher,
                                             Fjob_health, Fjob_other, Fjob_services, Fjob_teacher, reason_home,
                                             reason_other, reason_reputation, guardian_mother, guardian_other,
                                             schoolsup_yes, famsup_yes, paid_yes, activities_yes, nursery_yes,
                                             higher_yes, internet_yes, romantic_yes]])
            output = round(prediction[0])

        if output < 0:
            return render_template('index.html', prediction_texts="Sorry")
        else:
            return render_template('index.html', prediction_text="Absences = {}".format(output))
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
