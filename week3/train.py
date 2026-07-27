from ultralytics import YOLO


def main():
    # 사전 학습된 경량 모델을 불러온다.
    model = YOLO("yolov8n.pt")

    # data.yaml에 정의한 사용자 데이터셋으로 전이 학습을 수행한다.
    model.train(
        data="data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        project="runs",
        name="week3_yolov8n",
    )


if __name__ == "__main__":
    main()
