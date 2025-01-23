from fastapi import FastAPI 
from fastapi import FastAPI , File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import tensorflow as tf 

MODEL = tf.keras.models.load_model("model_v1.keras")
CLASS_NAMES = ['Early Blight','Late Blight','Healthy']

app = FastAPI()




@app.get("/ping")
async def ping():
  return "Hello yaswanth here"

def read_file_as_image(data) ->np.ndarray:
  image = np.array(Image.open(BytesIO(data)))
  return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  image = read_file_as_image(await file.read())
 
  img_bacth = np.expand_dims(image,0)
  prediction = MODEL.predict(img_bacth)
  
  predicted_class =  CLASS_NAMES[np.argmax(prediction[0])]
  confidence = np.max(prediction[0])
  
  return {
    'class' : predicted_class,
    'confidence' : float(confidence)
  }
  
  
  

if __name__ == "__main__":
  uvicorn.run(app,host='localhost',port=8000)
