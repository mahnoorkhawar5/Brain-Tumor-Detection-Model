from django.shortcuts import render

from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from django.conf import settings

# Load the trained CNN model once
model = load_model(r"D:\Artificual Intelligence\brain_tumor_cnn_model.h5")

def index(request):
    prediction = None
    if request.method == "POST" and request.FILES.get("image"):
        img_file = request.FILES['image']
        upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads', img_file.name)
        
        # Save uploaded image
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        with open(upload_path, 'wb+') as f:
            for chunk in img_file.chunks():
                f.write(chunk)

        # Preprocess for model
        img = load_img(upload_path, target_size=(64,64), color_mode="grayscale")
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension
        img_array = img_array[..., np.newaxis]           # channel dimension

        # Predict
        pred = model.predict(img_array)
        class_idx = np.argmax(pred, axis=1)[0]
        classes = ['Healthy', 'Tumor']
        prediction = classes[class_idx]

    return render(request, 'predictor/index.html', {'prediction': prediction})