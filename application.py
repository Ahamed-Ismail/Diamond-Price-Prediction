from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

@app.route('/test')
def test():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )

        final_data=data.get_data_as_dataframe()

        prediction_pipeline=PredictPipeline()
        
        pred=prediction_pipeline.predict(final_data)

        result=round(pred[0],2)

        return render_template('form.html', final_result=result)
    

if __name__=="__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)