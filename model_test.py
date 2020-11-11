from typing import *

import numpy as np
from PIL import Image

from environment_setting import test_dataset_path, test_model_weight_path
from face_recognizer import FaceRecognizer


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
            self.dataset_index = []
            for _ in range(10):
                fold = []
                for _ in range(300):
                    name, pic_num1, pic_num2 = lines[line_num].split("\t")
                    line_num += 1
                    face1_path = f"{test_dataset_path}{name}/{name}_{pic_num1.zfill(4)}.jpg"
                    face2_path = f"{test_dataset_path}{name}/{name}_{pic_num2.zfill(4)}.jpg"
                    fold.append((face1_path, face2_path, True))
                for _ in range(300):
                    name1, pic_num1, name2, pic_num2 = lines[line_num].split("\t")
                    line_num += 1
                    face1_path = f"{test_dataset_path}{name1}/{name1}_{pic_num1.zfill(4)}.jpg"
                    face2_path = f"{test_dataset_path}{name2}/{name2}_{pic_num2.zfill(4)}.jpg"
                    fold.append((face1_path, face2_path, False))
                self.dataset_index.append(fold)

    def _one_to_one_test(self, test_data: List[Tuple[str, str, bool]]):
        result = {}
        # 计算相似度
        similarities = []
        for face1_path, face2_path, label in test_data:
            similarity = self._get_similarity(face1_path, face2_path)
            similarities.append(similarity)
        # 计算准确率
        accuracy_rate, similarity_threshold = self._calculate_max_accuracy(
            similarities, (data[2] for data in test_data))
        # FRR @ FAR = x
        return accuracy_rate, similarity_threshold, similarities

    def make_one_to_one_test(self):
        result = []
        for i in range(10):
            result.append(self._one_to_one_test(self.dataset_index[i]))
        # 计算准确率
        test_num = 0
        correct_num = 0
        for i in range(10):
            similarity_threshold = result[i][1]
            for j in range(10):
                if i != j:
                    for similarity, (_, _, label) in zip(result[j][2], self.dataset_index[j]):
                        if (similarity >= similarity_threshold) == label:
                            correct_num += 1
                        test_num += 1
        correct_rate = correct_num / test_num
        # 计算FRR@FAR=
        return correct_rate

    @staticmethod
    def _calculate_max_accuracy(similarities, labels):
        similarities_with_labels = sorted(list(zip(similarities, labels)), key=lambda x: x[0])
        correct_num = sum(x[1] for x in similarities_with_labels)
        max_correct_num = correct_num
        similarity_threshold = 0
        for similarity, label in similarities_with_labels:
            if label == False:
                correct_num += 1
            else:
                correct_num -= 1
            if correct_num > max_correct_num:
                max_correct_num = correct_num
                similarity_threshold = similarity
        accuracy_rate = max_correct_num / len(similarities_with_labels)
        return accuracy_rate, similarity_threshold

    def one_to_more_test(self):
        pass

    def _get_face_feature(self, image_path):
        if image_path in self.features:
            return self.features[image_path]

        image = np.array(Image.open(image_path).convert('L'))
        feature = self.face_recognizer._get_feature_vector(image)
        self.features[image_path] = feature
        return feature

    def _get_similarity(self, face1_path, face2_path):
        face_pair = (face1_path, face2_path)
        if face_pair in self.similarities:
            return self.similarities[face_pair]

        feature1 = self._get_face_feature(face1_path)
        feature2 = self._get_face_feature(face2_path)
        similarity = self.face_recognizer._cosine_similarity(feature1, feature2)
        self.similarities[face_pair] = similarity
        return similarity


if __name__ == '__main__':
    lwf_tester = LFWTester()
    ac = lwf_tester.make_one_to_one_test()
