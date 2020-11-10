import os
import pickle
import unittest
from typing import *

import numpy as np
import torch
from PIL import Image
from numpy.linalg import norm
from torchvision import transforms

from environment_setting import device
from model import resnet_face18

__all__ = ['FaceRecognizer']


class FaceRecognizer:
    def __init__(self, model_weight_path: str, fingerprint_database_path: str = None) -> None:
        self.model = FaceRecognizer._load_model(model_weight_path)
        self.fingerprints = FaceRecognizer._load_fingerprint_database(fingerprint_database_path)
        self.fingerprint_database_path: str = fingerprint_database_path
        self.transforms = transforms.Compose([
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def face_recognize(self, image: np.ndarray) -> Tuple[str, float]:
        feature = self._get_feature_vector(image)
        similarity_max = 0
        name_of_max = ""
        for name, vector in self.fingerprints.items():
            similarity = FaceRecognizer._cosine_similarity(feature, vector)
            print(similarity, name)
            if similarity > similarity_max:
                name_of_max = name
                similarity_max = similarity
        return name_of_max, similarity_max

    def face_verification(self, image1, image2, threshold) -> Tuple[bool, float]:
        feature1 = self._get_feature_vector(image1)
        feature2 = self._get_feature_vector(image2)
        similarity = self._cosine_similarity(feature1, feature2)
        is_one_people = similarity >= threshold
        return is_one_people, similarity

    def add_fingerprint(self, images, names) -> None:
        for name, image in zip(names, images):
            feature = self._get_feature_vector(image)
            self.fingerprints[name] = feature

    def save_fingerprint_database(self) -> None:
        with open(self.fingerprint_database_path, 'wb') as f:
            pickle.dump(self.fingerprints, f)

    def _get_feature_vector(self, image) -> np.ndarray:
        image = self.transforms(Image.fromarray(image)).float()
        image = image.to(device)
        image = image.reshape([1, 1, 128, 128])
        feature_vector = self.model(image)
        return feature_vector.data.cpu().numpy()[0, :]

    @staticmethod
    def _load_model(path) -> torch.nn.Module:
        model = resnet_face18(use_se=False)
        model_weight = torch.load(path, map_location=device)
        model.load_state_dict(model_weight['backbone'])
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _load_fingerprint_database(path) -> Dict[str, np.ndarray]:
        if path is not None and os.path.exists(path):
            with open(path, 'rb') as f:
                fingerprints = pickle.load(f)
        else:
            fingerprints = {}
        return fingerprints

    @staticmethod
    def _cosine_similarity(feature1, feature2) -> float:
        return float(np.dot(feature1, feature2) / (norm(feature1) * norm(feature2)))


class FaceRecognizerTest(unittest.TestCase):
    def setUp(self) -> None:
        from environment_setting import test_dataset_path, run_model_weight_path, run_fingerprint_database_path
        self.face_recognizer = FaceRecognizer(run_model_weight_path, run_fingerprint_database_path)
        self.test_dataset_path = test_dataset_path

        img1_path = test_dataset_path + 'Adolfo_Rodriguez_Saa/Adolfo_Rodriguez_Saa_0001.jpg'
        img2_path = test_dataset_path + 'Adolfo_Rodriguez_Saa/Adolfo_Rodriguez_Saa_0002.jpg'
        img3_path = test_dataset_path + 'Adriana_Perez_Navarro/Adriana_Perez_Navarro_0001.jpg'
        self.image1 = np.array(Image.open(img1_path).convert('L'))
        self.image2 = np.array(Image.open(img2_path).convert('L'))
        self.image3 = np.array(Image.open(img3_path).convert('L'))

    def test_face_verification(self):
        """
        测试人脸验证 1:1
        """
        result = self.face_recognizer.face_verification(self.image1, self.image2, 0.3)
        print(result)
        self.assertTrue(result[0])
        result = self.face_recognizer.face_verification(self.image1, self.image3, 0.3)
        print(result)
        self.assertFalse(result[0])

    def test_face_recognizer(self):
        """
        测试人脸识别 1:N
        """
        images = [self.image1, self.image3]
        names = ['Adolfo_Rodriguez_Saa', 'Adriana_Perez_Navarro']
        self.face_recognizer.add_fingerprint(images, names)
        result = self.face_recognizer.face_recognize(self.image2)
        print(result)
        self.assertEqual(result[0], 'Adolfo_Rodriguez_Saa')
        self.face_recognizer.save_fingerprint_database()
        from environment_setting import run_model_weight_path, run_fingerprint_database_path
        face_recognizer = FaceRecognizer(run_model_weight_path, run_fingerprint_database_path)
        result = face_recognizer.face_recognize(self.image2)
        print(result)
        self.assertEqual(result[0], 'Adolfo_Rodriguez_Saa')


if __name__ == '__main__':
    unittest.main()
