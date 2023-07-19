from retinaface import RetinaFace
from PIL import Image
from torchvision import transforms
import warnings

class AgeRecognitionPreprocessor:
    def __init__(self, cust_transforms=None):
        self.default_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])  # ImageNet Normalization
            ])
        self.cust_transforms = cust_transforms
    
    def preprocess(self, image_path):
        pilim = Image.open(image_path)
        if self.cust_transforms:
            # Customized preprocessing steps
            return self.cust_transforms(pilim)
        else:
            return self.default_transforms(pilim)

class RetinaPreprocessor(AgeRecognitionPreprocessor):
    def __init__(self, output_size=(224, 224), cust_transforms=None):
        super(RecursionError, self).__init__(cust_transforms=cust_transforms)
        self.output_size = output_size
    
    def preprocess(self, image_path):
        detection = RetinaFace.detect_faces(image_path)
        pilim = Image.open(image_path)
        face = detection['face_1']
        face_crop = pilim.crop(face['facial_area'])
        if len(detection) > 1:
            warnings.warn(f'There is more than one face in the input image from {image_path}. Please ensure that the face below is the intended face for comparison:')
            face_crop.show()
        face_crop = face_crop.resize(self.output_size)
        if self.cust_transforms:
            # Customized preprocessing steps
            return self.cust_transforms(face_crop)
        else:
            return self.default_transforms(face_crop)


        