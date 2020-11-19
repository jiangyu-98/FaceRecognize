import os
from typing import *

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from environment import test_dataset_path, test_model_weight_path
from face_recognizer import FaceRecognizer


class LFWTester:
    def __init__(self):
        self._load_dataset_index()
        self.face_recognizer = FaceRecognizer(test_model_weight_path)
        self.features = {}
        self.similarities = {}

    def _load_dataset_index(self) -> None:
        # 加载1:1测试数据
        dataset_pairs = test_dataset_path + "pairs.txt"
        with open(dataset_pairs, "r") as f:
            lines = f.read().splitlines()
            line_num = 1
            self.dataset_one = []
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
                self.dataset_one.append(fold)

        # 加载1:N测试数据
        self.dataset_more_base, self.dataset_more_test = [], []
        for name in os.listdir(test_dataset_path):
            if os.path.isfile(test_dataset_path + name):
                continue
            pictures = os.listdir(test_dataset_path + name)
            for i, picture in enumerate(sorted(pictures)):
                if i == 0 and len(pictures) > 1:
                    self.dataset_more_test.append([picture, name])
                else:
                    self.dataset_more_base.append([picture, name])

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
            result.append(self._one_to_one_test(self.dataset_one[i]))
        # 计算准确率
        test_num = 0
        correct_num = 0
        for i in range(10):
            similarity_threshold = result[i][1]
            for j in range(10):
                if i != j:
                    for similarity, (_, _, label) in zip(result[j][2], self.dataset_one[j]):
                        if (similarity >= similarity_threshold) == label:
                            correct_num += 1
                        test_num += 1
        correct_rate = correct_num / test_num
        FRR_FAR_curve = []
        # 计算FRR(拒识率)@FAR(误识率)=x
        for threshold in np.arange(0, 1, 0.01):
            false_acceptance_num = 0
            false_rejection_num = 0
            true_acceptance_num = 0
            true_rejection_num = 0
            for i in range(10):
                for similarity, (_, _, label) in zip(result[i][2], self.dataset_one[i]):
                    if label == True:
                        if similarity >= threshold:
                            true_acceptance_num += 1
                        else:
                            false_acceptance_num += 1
                    else:
                        if similarity >= threshold:
                            false_rejection_num += 1
                        else:
                            true_rejection_num += 1
            FRR = false_rejection_num / (false_rejection_num + true_rejection_num)
            FAR = false_acceptance_num / (false_acceptance_num + true_acceptance_num)
            FRR_FAR_curve.append((FAR, FRR))
        return correct_rate, np.array(FRR_FAR_curve)

    @staticmethod
    def show_FRR_FAR_curve(FRR_FAR_curve: np.ndarray) -> None:
        plt.plot(FRR_FAR_curve[:, 0], FRR_FAR_curve[:, 1])

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

    def make_one_to_more_test(self):
        for i, (name, picture) in enumerate(self.dataset_more_base):
            self.face_recognizer.fingerprints[name + "?" + str(i)] = self._get_face_feature(picture)
        correct_num = 0
        for name, picture in self.dataset_more_test:
            feature = self._get_face_feature(picture)
            similarity_max = 0
            name_of_max = ""
            for name, vector in self.face_recognizer.fingerprints.items():
                similarity = FaceRecognizer._cosine_similarity(feature, vector)
                if similarity > similarity_max:
                    name_of_max = name
                    similarity_max = similarity
            if name_of_max.split("?")[0] == name:
                correct_num += 1
        return correct_num / len(self.dataset_more_test)

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
    # 1:1 测试
    # accuracy_rate, FRR_FAR_curve = lwf_tester.make_one_to_one_test()
    # print(accuracy_rate)
    # print(FRR_FAR_curve)
    # lwf_tester.show_FRR_FAR_curve(FRR_FAR_curve)

    # 1:N 测试
    lwf_tester.make_one_to_more_test()
