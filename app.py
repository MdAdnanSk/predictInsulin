from flask import Flask, render_template, request
import pickle as pkl
from ml.PredictInsulin import Mean_Absolute_Error, Mean_Squared_Error, Mean_Root_Squared_Error, accuracy


app=Flask(__name__)
model=pkl.load(open('model.pkl','rb'))

mae=round(Mean_Absolute_Error,2)
mse=round(Mean_Squared_Error,2)
mrse=round(Mean_Root_Squared_Error,2)
accu=round(accuracy,2)


@app.route("/")
def index():
    return render_template("index.html")
    
@app.route("/predict", methods=['POST'])
def predict():
    bloodglucose=float(request.form['bloodglucose'])
    exercise=float(request.form['exercise'])
    carbohydrateintake=float(request.form['carbohydrateintake'])
    prediction=model.predict([[bloodglucose,exercise,carbohydrateintake]])

    insulinunits=round(prediction[0],2)



    if insulinunits<0:
        insulin=f'Alert: Excessive exercise or low carb intake could lead to Hypogycemia'
    else:
        insulin=f'You Should inject {insulinunits} units of insulin'
        


    return render_template("predict.html",bloodglucose=f'Blood Glucose level={bloodglucose} mg/dl.',carbohydrateintake=f'Carbohydrate Intake={carbohydrateintake} gms.',exercise=f'Exercise ={exercise} hrs.', insulin=insulin, insulinunits=insulinunits)



@app.route("/scores")
def scores():
    return render_template("scores.html",mae=f'Mean Absolute Error={mae}',mse=f'Mean Squared Error={mse}',mrse=f'Mean Root Squared Error={mrse}',accu=f'Accuracy={accu}')


if __name__=="__main__":
    app.run()

