from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
import tensorflow as tf
import numpy as np

# Daftar label kategori
labels = ["Mobil", "Motor", "Bus"]

# Load model
model = tf.keras.models.load_model('deep_app/vehicle_classification_model.h5')

def home(request):
    return render(request, 'deep_app/index.html')  # Ini buat render halaman utama

def predict(request):
    if request.method == "POST" and request.FILES.get('file'):
        image_file = request.FILES['file']
        img = Image.open(image_file).resize((224, 224))  # Resize sesuai kebutuhan model
        img_array = np.array(img) / 255.0  # Normalisasi

        # Prediksi gambar
        prediction = model.predict(np.expand_dims(img_array, axis=0))[0]
        predicted_label = labels[np.argmax(prediction)]  # Ambil label sesuai prediksi

        # Kirim hasil prediksi ke template
        return render(request, 'deep_app/index.html', {'prediction': predicted_label})

    return render(request, 'deep_app/index.html')

