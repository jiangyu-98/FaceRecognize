import os
import random

import numpy as np
from matplotlib import pyplot as plt

import dataset
from environment import test_dataset_path, test_model_weight_path, cv2
from face_recognizer import FaceRecognizer

similarities = {}
features = {}


class LFWTester:
    def __init__(self):
        self._load_dataset_index()
        self.face_recognizer = FaceRecognizer(test_model_weight_path)

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
                    face1_path = f"{test_dataset_path}{name}/{name}_{pic_num1.zfill(4)}.png"
                    face2_path = f"{test_dataset_path}{name}/{name}_{pic_num2.zfill(4)}.png"
                    if os.path.exists(face1_path) and os.path.exists(face2_path):
                        fold.append((face1_path, face2_path, True))
                for _ in range(300):
                    name1, pic_num1, name2, pic_num2 = lines[line_num].split("\t")
                    line_num += 1
                    face1_path = f"{test_dataset_path}{name1}/{name1}_{pic_num1.zfill(4)}.png"
                    face2_path = f"{test_dataset_path}{name2}/{name2}_{pic_num2.zfill(4)}.png"
                    if os.path.exists(face1_path) and os.path.exists(face2_path):
                        fold.append((face1_path, face2_path, False))
                self.dataset_one.append(fold)

    def _load_dataset_more(self, max_size):
        # 加载1:N测试数据
        self.dataset_more_base, self.dataset_more_test = [], []
        for cnt, name in enumerate(os.listdir(test_dataset_path)):
            if os.path.isfile(test_dataset_path + name):
                continue
            if cnt > max_size:
                break
            pictures = os.listdir(test_dataset_path + name)
            pictures = [picture for picture in pictures if picture.endswith(".png")]
            for i, picture in enumerate(sorted(pictures)):
                face_path = f"{test_dataset_path}{name}/{picture}"
                if os.path.isfile(face_path):
                    if i == 0 and len(pictures) > 1:
                        self.dataset_more_test.append([face_path, name])
                    else:
                        self.dataset_more_base.append([face_path, name])

    def load_dataset_mask(self, test_num=2000):
        random.seed(444)
        images = dataset.get_mask_slim_datalist()
        data = dict()
        for image_path, label in images:
            if label in data:
                data[label].append(image_path)
            else:
                data[label] = [image_path]
        self.dataset_one = []
        idx = list(data.keys())
        for i in range(int(test_num / 2)):
            p1 = random.choice(idx)
            p2 = p1
            while p2 == p1:
                p2 = random.choice(idx)
            self.dataset_one.append((
                random.choice(data[p1]),
                random.choice(data[p2]),
                False
            ))
        for i in range(int(test_num / 2)):
            p1 = random.choice(idx)
            while len(data[p1]) == 1:
                p1 = random.choice(idx)
            i1 = random.choice(data[p1])
            i2 = i1
            while i2 == i1:
                i2 = random.choice(data[p1])

            self.dataset_one.append((
                i1,
                i2,
                True
            ))

    def _one_to_one_test(self, test_data):
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

    def make_one_to_more_test(self, max_size):
        self._load_dataset_more(max_size)
        for i, (picture, name) in enumerate(self.dataset_more_base):
            # print(picture)
            try:
                self.face_recognizer.fingerprints[name + "?" + str(i)] = self._get_face_feature(picture)
            except:
                pass
        # print("ok")
        correct_num = 0
        cnt = 0
        for picture, name in self.dataset_more_test:
            # print(f"{cnt} / {len(self.dataset_more_test)}")
            cnt += 1
            feature = self._get_face_feature(picture)
            similarity_max = 0
            name_of_max = ""
            for p, vector in self.face_recognizer.fingerprints.items():
                similarity = FaceRecognizer._cosine_similarity(feature, vector)
                if similarity > similarity_max:
                    name_of_max = p
                    similarity_max = similarity
            # print(f"{similarity_max}, {nam    e_of_max}, {p}")
            if name_of_max.startswith(name):
                correct_num += 1
        return correct_num / len(self.dataset_more_test)

    @staticmethod
    def show_FRR_FAR_curve(FRR_FAR_curve: np.ndarray) -> None:
        plt.plot(FRR_FAR_curve[:, 0], FRR_FAR_curve[:, 1])
        plt.xlabel("FAR")
        plt.ylabel("FRR")
        plt.show()

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

    def _get_face_feature(self, image_path):
        if image_path in features:
            return features[image_path]

        image = cv2.imread(image_path)
        feature = self.face_recognizer._get_feature_vector(image)
        features[image_path] = feature
        return feature

    def _get_similarity(self, face1_path, face2_path):
        face_pair = (face1_path, face2_path)
        if face_pair in similarities:
            return similarities[face_pair]

        feature1 = self._get_face_feature(face1_path)
        feature2 = self._get_face_feature(face2_path)
        similarity = self.face_recognizer._cosine_similarity(feature1, feature2)
        similarities[face_pair] = similarity
        return similarity


if __name__ == '__main__':
    lfw_tester = LFWTester()
    # data = lfw_tester.load_dataset_mask(2000)
    # accuracy_rate, similarity_threshold, similarities = lfw_tester._one_to_one_test(lfw_tester.dataset_one)
    # print(accuracy_rate)

    # # 1:1 测试
    accuracy_rate, FRR_FAR_curve = lfw_tester.make_one_to_one_test()
    print(accuracy_rate)
    print(FRR_FAR_curve)
    lfw_tester.show_FRR_FAR_curve(FRR_FAR_curve)

    # 1:N 测试
    # for i in range(1000, 14000, 2000):
    #     acc = lfw_tester.make_one_to_more_test(i)
    #     print(f"{i}, {acc}")
