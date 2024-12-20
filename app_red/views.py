import torch
import torchvision.transforms as transforms
from PIL import Image
from django.shortcuts import render
from appCNN.settings import BASE_DIR

MODEL_PATH = BASE_DIR / 'static/modelo' / 'custom_cnn_scripted.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
try:
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    print("Modelo cargado correctamente.")

except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def upload(request):
    return render(request, 'upload.html')


# Vista para predecir
def predict(request):
    if request.method == 'POST':
        if model is None:
            return render(request, 'upload.html', {'error': 'Error al cargar el modelo. Por favor, intenta nuevamente.'})

        image = request.FILES['image']
        image = Image.open(image).convert("L")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        resultado = "Pneumonia" if predicted.item() == 1 else "Saludable"
        return render(request, 'result.html', {'resultado': resultado})
    return render(request, 'upload.html')
