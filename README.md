# 🍼 영유아 행동 인식 시스템 — ViViT 기반

> **Video Vision Transformer(ViViT)** 를 활용하여 영유아의 발달 행동을 자동으로 인식·분류하는 프로젝트입니다.  
> 어린이집 교사가 원아의 행동을 빠짐없이 기록하기 어렵다는 현장의 문제에서 출발했습니다.

---

## 📋 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **기간** | 2023년 |
| **팀** | 고려대 데청캠 12조 (5인) |
| **목표** | 영유아 행동 영상 → 자동 행동 분류 |
| **모델** | ViViT (Video Vision Transformer) |
| **최종 성능** | Test Accuracy **86.7%** |

---

## 🎯 주제 선정 배경

어린이집 표준보육과정(제4차)에 따르면, 교사는 원아의 행동 발달을 6개 영역으로 관찰·기록해야 합니다.

- **01 기본 생활** / **02 신체 운동** / **03 의사소통**
- **04 사회 관계** / **05 예술 경험** / **06 자연 탐구**

현장 인터뷰 결과, *"물리적으로 모든 행동을 기록하기 어렵다"* 는 교사들의 공통된 어려움과 *"우리 아이의 발달 정보를 더 많이 받고 싶다"* 는 학부모의 니즈를 확인했습니다.

---

## 🔍 분류 대상 행동 (3 Classes)

| 라벨 | 행동 | 발달 영역 |
|------|------|-----------|
| `Clapping` | 손뼉 치기 | 신체운동 — 신체조절과 기본운동하기 |
| `Sipping` | 컵으로 마시기 | 기본생활 — 건강하게 생활하기 |
| `Stacking_Rings` | 고리 쌓기 | 신체운동 — 감각과 신체 인식하기 |

---

## 🗂️ 데이터셋

### 수집 방법
- **Clapping / Sipping**: DeepMind Kinetics-600 + YouTube 크롤링
- **Stacking Rings**: Kaggle + YouTube 크롤링

### 전처리 파이프라인

```
원본 영상
  → 10프레임 간격 추출 (30FPS 기준: 초당 3장)
  → 해상도 조정 (60×60)
  → 영상당 12프레임 확보
  → train(70%) / val(15%) / test(15%) 분할
  → .npz 파일로 저장 (shape: [N, 12, 60, 60, 3])
```

---

## 🏗️ 모델 아키텍처 — ViViT

이미지 분류에 특화된 **Vision Transformer(ViT)** 를 비디오 데이터로 확장한 모델입니다.

```
입력 (12, 60, 60, 3)
    ↓
Tubelet Embedding (Conv3D, patch_size=4×4×4)
    ↓
Positional Embedding
    ↓
Transformer Encoder × 8 (Spatio-Temporal Attention)
    ↓
Global Average Pooling
    ↓
Dense + Softmax → 3 Classes
```

### 핵심 컴포넌트

**1. Tubelet Embedding**
- 3D Conv로 시공간 패치(tube) 추출
- ViT의 패치 임베딩을 시간 차원으로 확장

**2. Spatio-Temporal Attention**
- 공간(Spatial): 프레임 내 중요 영역 집중 (얼굴 < 손)
- 시간(Temporal): 중요 프레임 집중 (고리 찾기 < 고리 쌓기)

**3. MLP Multi-Head Attention (8 heads)**
- 토큰 간 복잡한 관계 포착
- 다양한 종속성을 병렬로 처리하여 표현력 향상

### 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| Input Shape | (12, 60, 60, 3) |
| Patch Size | (4, 4, 4) |
| Projection Dim | 128 |
| Num Heads | 8 |
| Num Layers | 8 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| Batch Size | 16 |
| Epochs | 60 |

---

## 📊 실험 결과

| 지표 | 값 |
|------|-----|
| Test Accuracy | **86.7%** |
| Test Top-5 Accuracy | 100.0% |

### 하이퍼파라미터 탐색 결과
- **Epoch**: 20 이후 성능 수렴 → 60으로 설정
- **Batch Size**: 16일 때 가장 효율적 (32→16→8→4 순으로 급격히 하락)
- **Patch Size**: 이미지 크기(60×60)를 고려해 4×4×4로 설정

### 한계점 및 개선 방향
- 소규모 데이터셋으로 인한 과적합 위험
- 특정 구역(손, 얼굴) 기반의 명시적 탐지 없이 전체 프레임에서 분류 → attention 기반의 암묵적 focusing에 의존
- 향후 개선: 객체 탐지(YOLO 등) 결합으로 특정 신체 부위에 집중한 분류 시도 가능

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
git clone https://github.com/YOUR_USERNAME/vivit-child-action.git
cd vivit-child-action
pip install -r requirements.txt
```

### 2. 데이터 수집

```bash
# YouTube에서 영상 크롤링
jupyter notebook crawling.ipynb
```

수집한 영상은 아래 디렉토리 구조로 저장하세요:

```
action/
├── Clapping/
│   ├── video1.mp4
│   └── ...
├── Sipping/
│   └── ...
└── Stacking_Rings/
    └── ...
```

### 3. 전처리

```bash
jupyter notebook preprocess.ipynb
```

전처리 완료 후 `inputdata/` 폴더에 아래 파일이 생성됩니다:
```
inputdata/
├── train_with_labels_second.npz
├── valid_with_labels_second.npz
└── test_with_labels_second.npz
```

### 4. 모델 학습 및 추론

```bash
jupyter notebook model.ipynb
```

---

## 📁 파일 구조

```
vivit-child-action/
├── README.md
├── requirements.txt
├── parameters.txt          # 하이퍼파라미터 설정
├── labels.txt              # 클래스 라벨 정의
├── crawling.ipynb          # 유튜브 영상 크롤링
├── preprocess.ipynb        # 데이터 전처리 및 분할
└── model.ipynb             # ViViT 모델 정의, 학습, 추론
```

---

## 🔮 서비스 기획 구상

1. **발달 특성 체크리스트 자동 작성**: 탐지된 행동을 날짜·아동별로 자동 분류 및 기록
2. **우리 아이 행동 알리미**: 탐지 이미지 + 활동 설명 문장 생성 → 학부모 앱 알림

### 기대 효과
- 교사가 놓치는 사소한 행동까지 자동 체크
- 발달 지연 조기 발견 가능
- 부모에게 더 풍부한 발달 정보 제공

### 발전 방향
- 다중 객체 탐지(Multiple Object Detection)로 여러 아이 동시 인식
- 자연어 모델(LLM) 결합 → 행동 설명 문장 자동 생성
- 가정용 앱으로 확장

---

## 📚 참고 문헌

- [ViViT: A Video Vision Transformer (ICCV 2021)](https://arxiv.org/abs/2103.15691)
- [An Image is Worth 16x16 Words: ViT (ICLR 2021)](https://arxiv.org/abs/2010.11929)
- [Is Space-Time Attention All You Need for Video Understanding? (ICML 2021)](https://arxiv.org/abs/2102.05095)
- [보건복지부 고시 제2020-75호 — 제4차 표준보육과정](https://www.mohw.go.kr)
- [TensorFlow ViViT Tutorial](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub)
