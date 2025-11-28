import cv2
import os
import numpy as np

# model(이미지)의 리턴값인 results[0] = result, 처리한 이미지가 filename(sample_test/image1, image2, image3...)
def crop_by_result(result, filename, output_dir='output_crops'):
    orig_img = result.orig_img
    h, w, _ = orig_img.shape

    # 마스크나 박스가 없으면 종료
    if result.masks is None:
        print("탐지된 객체가 없습니다.")
        return

    # 2. 각 객체별로 반복 처리
    # result.boxes: 바운딩 박스 정보
    # result.masks.data: 마스크 데이터 (GPU 텐서일 수 있음)

    masks = result.masks.data.cpu().numpy() # 텐서를 넘파이 배열로 변환
    boxes = result.boxes.xyxy.cpu().numpy() # 박스 좌표 (x1, y1, x2, y2)
    cls_ids = result.boxes.cls.cpu().numpy() # 클래스 ID

    for i, mask in enumerate(masks):
        # --- [Step A] 마스크 크기 맞추기 ---
        # YOLO 마스크는 보통 640x640 등의 추론 크기로 나오므로 원본 크기로 리사이징 필요
        # cv2.resize(src, dsize=(width, height))
        mask = mask.astype(np.uint8)
        resized_mask = cv2.resize(mask, (w, h))

        # 마스크 이진화 (0 아니면 1로 확실하게 구분)
        # 보통 0.5 이상을 객체로 판단
        binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255

        # --- [Step B] 투명 배경 이미지 만들기 (BGRA) ---
        # 원본 이미지(BGR) 채널 분리
        b, g, r = cv2.split(orig_img)

        # 알파 채널로 마스크 사용 (배경은 0=투명, 객체는 255=불투명)
        rgba_img = cv2.merge([b, g, r, binary_mask])

        # --- [Step C] Bounding Box로 Crop ---
        x1, y1, x2, y2 = boxes[i].astype(int)

        # 좌표가 이미지 범위를 벗어나지 않도록 클리핑
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        # 이미지 자르기 (y축 먼저, x축 나중)
        crop_img = rgba_img[y1:y2, x1:x2]

        # --- [Step D] 저장 ---
        class_name = result.names[int(cls_ids[i])]
        # file_name = f"{class_name}_{i}.png" # 투명도 저장을 위해 반드시 png 사용
        # 2. 확장자 분리 ('test.jpg' -> 'test', '.jpg')

        # 3. 파일명 생성 ('test_chair_0.png')
        file_name = f"{class_name}/{filename}_{i}.png"

        # 4. 저장 경로 결합
        save_path = os.path.join(output_dir, file_name)
        cv2.imwrite(save_path, crop_img)
        print(f"[저장 완료] {save_path} (크기: {crop_img.shape})")