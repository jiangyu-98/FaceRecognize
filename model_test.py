from face_recognizer import FaceRecognizer
from environment_setting import test_dataset_path, test_model_weight_path
from PIL import Image
import numpy as np
from typing import *


class LFWTester:
    def __init__(self):
        self._load_dataset_index()
        self.face_recognizer = FaceRecognizer(test_model_weight_path)
        self.features = {}
        self.similarities = {}

    def _load_dataset_index(self) -> None:
        dataset_pairs = test_dataset_path + "pairs.txt"
        with open(dataset_pairs, "r") as f:
            lines = f.read().splitlines()
            line_num = 1
            self.dataset_pairs = []
            for _ in range(10):
                for _ in range(300):
                    fold = []
                    name, pic_num1, pic_num2 = lines[line_num].split("\t")
                    line_num += 1
                    face1_path = f"{test_dataset_path}{name}/{name}_{pic_num1.zfill(4)}"
                    face2_path = f"{test_dataset_path}{name}/{name}_{pic_num2.zfill(4)}"
                    fold.append((face1_path, face2_path, True))
                for _ in range(300):
                    name1, pic_num1, name2, pic_num2 = lines[line_num].split("\t")
                    line_num += 1
                    face1_path = f"{test_dataset_path}{name1}/{name1}_{pic_num1.zfill(4)}"
                    face2_path = f"{test_dataset_path}{name2}/{name2}_{pic_num2.zfill(4)}"
                    fold.append((face1_path, face2_path, False))
                self.dataset_pairs.append(fold)

    def one_to_one_test(self, test_data: List[str, str, bool]):
        similarities = []
        for face1_path, face2_path, _ in test_data:
            similarity = self.get_similarity(face1_path, face2_path)
            similarities.append(similarity)

    def one_to_more_test(self):
        pass

    def get_face_feature(self, image_path):
        if image_path in self.features:
            return self.features[image_path]

        image = np.array(Image.open(image_path).convert('L'))
        feature = self.face_recognizer._get_feature_vector(image)
        self.features[image_path] = feature
        return feature

    def get_similarity(self, face1_path, face2_path):
        face_pair = (face1_path, face2_path)
        if face_pair in self.similarities:
            return self.similarities[face_pair]

        feature1 = self.get_face_feature(face1_path)
        feature2 = self.get_face_feature(face2_path)
        similarity = self.face_recognizer._cosine_similarity(feature1, feature2)
        self.similarities[face_pair] = similarity
        return similarity


if __name__ == '__main__':
    lwf_tester = LFWTester()
