import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import face_alignment
import numpy as np
import cv2

mean_face_shape_x = np.array([0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                              0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                              0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                              0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                              0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                              0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                              0.553364, 0.490127, 0.42689])

mean_face_shape_y = np.array([0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                              0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                              0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                              0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                              0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                              0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                              0.784792, 0.824182, 0.831803, 0.824182])
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')


class FaceAligner:
    def __init__(self, size=224, padding=0.25):
        x = ((mean_face_shape_x + padding) / (2 * padding + 1)) * size
        y = ((mean_face_shape_y + padding) / (2 * padding + 1)) * size
        self.front = np.dstack((x, y))[0]
        self.size = 224

    def align(self, image: np.ndarray) -> np.ndarray:
        point = fa.get_landmarks(image)[0]
        affine_matrix = self.transformation_from_points(np.matrix(point[17:]), np.matrix(self.front))
        return cv2.warpAffine(image, affine_matrix[:2], (self.size, self.size))

    @staticmethod
    def transformation_from_points(points1, points2):
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= np.mean(points1, axis=0)
        points2 -= np.mean(points2, axis=0)
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2
        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


if __name__ == '__main__':
    image = cv2.imread('0003_01.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_aligner = FaceAligner()
    image = face_aligner.align(image)
    # cv2.imshow(image)
    from PIL import Image

    Image.fromarray(image).show(image)
