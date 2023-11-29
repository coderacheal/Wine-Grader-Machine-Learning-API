from fastapi import FastAPI, HTTPException, Response
import joblib
from pydantic import BaseModel
import pandas as pd
import uvicorn

pipeline = joblib.load('./model/pipeline.joblib')
encoder = joblib.load('./model/encoder.joblib')

app = FastAPI()

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float 
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float 
    color_intensity: float 
    hue: float
    od280_od315_of_diluted_wines: float 
    proline: float


@app.get('/')
def home():
    return Response('Hello world')


@app.get('/info')
def appinfo():
    return Response('This is the info page of this API')


@app.post('/predict_grade')
def predict_wine_grade(wine_features: WineFeatures):

    try: 
    
        df = pd.DataFrame([wine_features.model_dump()])

        prediction = pipeline.predict(df)

        decoded_prediction = encoder.inverse_transform([prediction])[0]

        return {'prediction': decoded_prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'This is a server error {str(e)}')


# if __name__ == '__main__':
#     uvicorn.run(a)  