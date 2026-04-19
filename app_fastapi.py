from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

clf_model = joblib.load('artifacts/placement_classification.pkl')
reg_model = joblib.load('artifacts/salary_regression.pkl')

class DataInput(BaseModel):
    # Fitur Kategorikal (object)
    gender: str
    extracurricular_activities: str
    
    # Fitur Numerik (int64 & float64)
    ssc_percentage: int
    hsc_percentage: int
    degree_percentage: int
    cgpa: float
    entrance_exam_score: int
    technical_skill_score: int
    soft_skill_score: int
    internship_count: int
    live_projects: int
    work_experience_months: int
    certifications: int
    attendance_percentage: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Placement and Salary Prediction"}

# def fitur_baru(df):    
#     academic_index = (df['ssc_percentage'].iloc[0] + 
#                       df['hsc_percentage'].iloc[0] + 
#                       df['degree_percentage'].iloc[0] + 
#                       (df['cgpa'].iloc[0] * 10)) / 4
                      
#     job_readiness = (df['internship_count'].iloc[0] + 
#                      df['live_projects'].iloc[0] + 
#                      df['certifications'].iloc[0])
                     
#     total_comp = (df['technical_skill_score'].iloc[0] + 
#                   df['soft_skill_score'].iloc[0])

#     return round(academic_index, 2), int(job_readiness), int(total_comp)

@app.post('/predict')
def predict(people: DataInput):
    # Mengubah data masuk dari bentuk json (pydantic payload) yang kemudian di validasi oleh pydantic menjadi bentuk Pandas Dataframe
    # Dictionary keys otomatis akan jadi nama kolomnya di dataframe
    df = pd.DataFrame([people.model_dump()])

    # aca_idx, job_ready, comp_score = fitur_baru(df)
    
    prediction = clf_model.predict(df)[0]

    status_map = {0: 'Not Placed', 1: 'Placed'}
    status_text = status_map.get(int(prediction))

    if prediction == 1:
        salary_pred = reg_model.predict(df)[0]
        final_salary = round(float(max(0, salary_pred)), 2)
    else :
        final_salary = 0.0
    
    # Dijadikan int karena JSON hanya bisa menerima int dari python dan bukan yg lain (numpy.int64) untuk melakukan JSON Serialization
    return {
        'prediction_code' : int(prediction),
        'prediction_label': status_text,
        'estimated_salary_lpa' : final_salary,
        'currency' : "Lakhs Per Annum"
        # "new_feats" : {
        #     "academic_index": aca_idx,
        #     "job_readiness": job_ready,
        #     "total_competency": comp_score
        # }
    }