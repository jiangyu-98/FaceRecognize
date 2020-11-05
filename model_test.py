from face_recognizer import FaceRecognizer
from environment_setting import test_dataset_path, test_model_weight_path


class LFWTester:
    def __init__(self):
        self._load_dataset_index()
        self.face_recognizer = FaceRecognizer(test_model_weight_path)
        self.features = {}

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
                    path1 = f"{test_dataset_path}{name}/{name}_{pic_num1.zfill(4)}"
                    path2 = f"{test_dataset_path}{name}/{name}_{pic_num2.zfill(4)}"
                    fold.append((path1, path2, True))
                for _ in range(300):
                    name1, pic_num1, name2, pic_num2 = lines[line_num].split("\t")
                    line_num += 1
                    path1 = f"{test_dataset_path}{name1}/{name1}_{pic_num1.zfill(4)}"
                    path2 = f"{test_dataset_path}{name2}/{name2}_{pic_num2.zfill(4)}"
                    fold.append((path1, path2, False))
                self.dataset_pairs.append(fold)

    def make_test(self, data):
        pass

    def get_face_feature(self, image_path):
        if image_path in self.features:
            return self.features[image_path]


if __name__ == '__main__':
    lwf_tester = LFWTester()
