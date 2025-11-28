from ultralytics.models.sam import SAM2DynamicInteractivePredictor
from ultralytics import YOLOE
from crop_by_result import crop_by_result
import os

def crop():
    # Create SAM2DynamicInteractivePredictor
    overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_t.pt", save=False)
    predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)

    model = YOLOE('yoloe-11l-seg.pt')
    names = [
        "Kitchen Cabinet",              # 700292: 주방 캐비닛
        "Mini Kitchen",                 # 22957: 미니 주방
        "Kitchen Island/Cart",          # 10471: 아일랜드 식탁/주방 카트
        "Kitchen Appliance",            # ka002: 주방 가전
        "Kitchen Countertop",           # 24264: 주방 조리대
        "Kitchen Pantry",               # 16200: 주방 팬트리
        "Kitchen System",               # ka003: 주방 시스템
        "Office Desk/Chair Set",        # 700424: 오피스 책상/의자 세트
        "Conference Chair",             # 47068: 회의실 의자
        "Gaming Furniture",             # 55002: 게임용 가구
        "Conference Table",             # 54173: 회의 테이블
        "Desk/Chair Set",               # 53249: 책상/의자 세트
        "Office Chair",                 # 20652: 의자/사무실 의자
        "Computer Desk",                # 20649: 책상/컴퓨터 책상
        "Vanity Chair/Stool",           # 59250: 화장대 의자/스툴
        "Toddler Chair",                # 45782: 영유아 의자
        "Childrens Chair",              # bc004: 어린이 의자
        "Childrens Table",              # 18768: 어린이 테이블
        "Step Stool",                   # 20611: 스텝 스툴/스텝 의자
        "Bench",                        # 700319: 벤치
        "Cafe Furniture",               # 19141: 카페 가구
        "Stool",                        # 22659: 스툴
        "Bar Table/Chair",              # 16244: 바 테이블/의자
        "Coffee/Side Table",            # 10705: 커피테이블/사이드테이블
        "Chair",                        # 700676: 의자
        "Table",                        # 700675: 테이블
        "Dining Furniture",             # 700417: 다이닝 가구
        "Chaise Longue/Couch",          # 57527: 긴 의자/카우치
        "Footstool",                    # 20926: 풋스툴/발받침대
        "Sofa Bed",                     # 10663: 소파침대/소파베드
        "Armchair",                     # fu006: 암체어/안락의자
        "Sofa",                         # fu003: 소파
        "Bedroom Set",                  # 54992: 침실 가구 세트
        "Bed with Mattress",            # 700513: 매트리스 포함 침대
        "Bedside Table",                # 20656: 침대 협탁
        "Bed Frame",                    # bm003: 침대/침대프레임
        "Shoe Cabinet",                 # 10456: 신발장
        "Storage Unit",                 # 10385: 수납유닛/수납장
        "Toy Storage",                  # 20474: 장난감 수납/정리함
        "Hallway Set",                  # 700411: 복도 가구 세트
        "Partition",                    # 46080: 칸막이/파티션
        "Drawer/Nightstand",            # st004: 서랍장/침대협탁
        "Storage System",               # 46052: 수납 솔루션 시스템
        "Sideboard/Console Table",      # 30454: 거실장/찬장/콘솔테이블
        "Trolley",                      # fu005: 트롤리/미니 카트
        "TV/Media Furniture",           # 10475: TV/멀티미디어 가구
        "Outdoor Storage",              # 21958: 야외용 수납가구
        "Warehouse Storage",
        "cabinet"           # 700440: 창고 수납
    ]

    img_dir = 'sample_test'

    imgs = os.listdir(img_dir)
    print(imgs)

    for i, img in enumerate(imgs):
        imgsrc = os.path.join(img_dir, img)
        if i==0:
            model.set_classes(names, model.get_text_pe(names))
            results = model.predict(imgsrc)
            predictor(source=imgsrc, bboxes=results[0].boxes.xyxy.cpu().numpy(), obj_ids=[i for i in range(len(results[0].boxes))], update_memory=True)
            results = predictor(source=imgsrc)
            for i in range(len(results[0].boxes)):
                os.makedirs('output_crops/'+str(i))
            crop_by_result(results[0], img)
        else:
            results = predictor(source=imgsrc)
            crop_by_result(results[0], img)

