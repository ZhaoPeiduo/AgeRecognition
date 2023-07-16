from retinaface import RetinaFace
from PIL import Image
from torchvision import transforms
import warnings

class AgeRecognitionPreprocessor:
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
    
    def preprocess(self, image_path, cust_transforms=None):
        detection = RetinaFace.detect_faces(image_path)
        pilim = Image.open(image_path)
        face = detection['face_1']
        face_crop = pilim.crop(face['facial_area'])
        if len(detection) > 1:
            warnings.warn(f'There is more than one face in the input image from {image_path}. Please ensure that the face below is the intended face for comparison:')
            face_crop.show()
        face_crop = face_crop.resize(self.output_size)
        if cust_transforms:
            # Customized preprocessing steps
            return cust_transforms(face_crop)
        else:
            default_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])  # ImageNet Normalization
            ])
            return default_transforms(face_crop)


        