"""학습된 YOLO 분류 모델로 은하 이미지를 예측하고 시각화하는 모듈."""
import cv2

TEXT_BG = (77, 72, 229)   # BGR — 시그니처 레드
TEXT_COLOR = (255, 255, 255)


class GalaxyClassifier:
    """학습된 YOLO 분류 모델 래퍼. model 주입으로 단위 테스트 용이."""

    def __init__(self, model=None, model_path="runs/classify/train/weights/best.pt"):
        self._model = model
        self._model_path = model_path

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
        return self._model

    def predict(self, image, topk=3):
        """이미지 1장의 은하 형태를 예측한다.

        Returns:
            {"label": str, "confidence": float, "topk": [(label, conf), ...]}
        """
        if image is None:
            raise ValueError("입력 이미지가 없습니다.")
        if topk <= 0:
            raise ValueError("topk는 1 이상이어야 합니다.")

        result = self.model(image, verbose=False)[0]
        probs = result.probs
        names = result.names
        top_indices = probs.top5[:topk]
        pairs = [(names[i], float(probs.data[i])) for i in top_indices]
        return {"label": pairs[0][0], "confidence": pairs[0][1], "topk": pairs}


def annotate(image, prediction):
    """예측 결과(라벨 + confidence)를 이미지 좌상단에 그린다. 원본 비파괴."""
    if image is None:
        raise ValueError("입력 이미지가 없습니다.")
    if prediction is None:
        raise ValueError("예측 결과가 None입니다.")

    annotated = image.copy()
    label = f'{prediction["label"]} {prediction["confidence"]:.2f}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated, (8, 8), (16 + tw, 20 + th), TEXT_BG, -1)
    cv2.putText(annotated, label, (12, 14 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
    return annotated
