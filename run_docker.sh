#!/bin/bash
# ============================================================
# run_docker.sh - Docker 컨테이너 관리 스크립트
# ============================================================
#
# SSH 연결이 끊겨도 컨테이너는 계속 실행됩니다.
# (docker-compose.yml의 restart: unless-stopped 설정)
#
# 사용법:
#   ./run_docker.sh build     # 이미지 빌드
#   ./run_docker.sh up        # 컨테이너 시작
#   ./run_docker.sh down      # 컨테이너 중지
#   ./run_docker.sh exec      # 컨테이너 접속 (bash)
#   ./run_docker.sh train     # 학습 실행 (백그라운드)
#   ./run_docker.sh logs      # 로그 확인
#   ./run_docker.sh status    # 상태 확인
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
CONTAINER_NAME="banner_project"

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ── 명령어 처리 ────────────────────────────────────────────
case "$1" in

    # ── 이미지 빌드 ──────────────────────────────────────────
    build)
        print_info "Docker 이미지 빌드 중..."
        print_info "  (처음 빌드는 10~20분 소요)"
        docker compose -f "${COMPOSE_FILE}" build
        print_success "빌드 완료"
        ;;

    # ── 컨테이너 시작 ────────────────────────────────────────
    up)
        print_info "컨테이너 시작 중..."
        docker compose -f "${COMPOSE_FILE}" up -d
        print_success "컨테이너 실행 중"
        echo ""
        echo "  SSH 연결이 끊겨도 컨테이너는 계속 실행됩니다."
        echo "  컨테이너 접속: ./run_docker.sh exec"
        ;;

    # ── 컨테이너 중지 ────────────────────────────────────────
    down)
        print_warn "컨테이너를 중지합니다..."
        docker compose -f "${COMPOSE_FILE}" down
        print_success "컨테이너 중지됨"
        ;;

    # ── 컨테이너 접속 ────────────────────────────────────────
    exec | shell)
        print_info "컨테이너 접속 중..."
        docker exec -it "${CONTAINER_NAME}" bash
        ;;

    # ── 학습 실행 (Docker 내부에서 백그라운드로) ─────────────
    train)
        print_info "학습 시작 (백그라운드)..."
        # 컨테이너 내부에서 nohup으로 실행
        docker exec -d "${CONTAINER_NAME}" bash -c "
            cd /workspace/project && \
            nohup bash -c '
                python 01_prepare_data.py 2>&1 | tee /workspace/results/log_prepare.txt && \
                python 02_train_yolo.py   2>&1 | tee /workspace/results/log_train.txt && \
                python 03_evaluate_yolo.py 2>&1 | tee /workspace/results/log_eval.txt
            ' > /workspace/results/log_full.txt 2>&1
        "
        print_success "학습이 백그라운드에서 시작되었습니다"
        echo ""
        echo "  로그 확인:"
        echo "    ./run_docker.sh logs"
        echo "  또는:"
        echo "    docker exec ${CONTAINER_NAME} tail -f /workspace/results/log_full.txt"
        ;;

    # ── 데이터 준비만 실행 ────────────────────────────────────
    prepare)
        print_info "데이터 준비 실행..."
        docker exec -it "${CONTAINER_NAME}" \
            python /workspace/project/01_prepare_data.py
        ;;

    # ── YOLO 학습만 실행 ─────────────────────────────────────
    yolo-train)
        EPOCHS=${2:-100}
        print_info "YOLO 학습 (${EPOCHS} 에폭)..."
        docker exec -it "${CONTAINER_NAME}" \
            python /workspace/project/02_train_yolo.py --epochs "${EPOCHS}"
        ;;

    # ── 평가 실행 ────────────────────────────────────────────
    eval)
        print_info "모델 평가..."
        docker exec -it "${CONTAINER_NAME}" \
            python /workspace/project/03_evaluate_yolo.py
        ;;

    # ── 파이프라인 데모 ───────────────────────────────────────
    demo)
        IMAGE=${2:-""}
        if [ -z "$IMAGE" ]; then
            print_error "이미지 경로를 지정하세요: ./run_docker.sh demo /path/to/image.jpg"
            exit 1
        fi
        print_info "파이프라인 데모: ${IMAGE}"
        docker exec -it "${CONTAINER_NAME}" \
            python /workspace/project/04_pipeline.py \
                --image "${IMAGE}" --visualize \
                --output /workspace/results/pipeline_output
        ;;

    # ── 로그 확인 ────────────────────────────────────────────
    logs)
        print_info "학습 로그 (Ctrl+C로 종료):"
        docker exec "${CONTAINER_NAME}" \
            tail -f /workspace/results/log_full.txt 2>/dev/null || \
        docker compose -f "${COMPOSE_FILE}" logs -f
        ;;

    # ── 상태 확인 ────────────────────────────────────────────
    status)
        print_info "컨테이너 상태:"
        docker compose -f "${COMPOSE_FILE}" ps
        echo ""
        print_info "GPU 사용량:"
        docker exec "${CONTAINER_NAME}" nvidia-smi 2>/dev/null || \
            print_warn "GPU 정보를 가져올 수 없습니다"
        ;;

    # ── 도움말 ───────────────────────────────────────────────
    help | --help | -h | "")
        echo "사용법: ./run_docker.sh <명령>"
        echo ""
        echo "명령:"
        echo "  build        Docker 이미지 빌드 (최초 1회)"
        echo "  up           컨테이너 시작 (백그라운드)"
        echo "  down         컨테이너 중지"
        echo "  exec         컨테이너 bash 접속"
        echo "  train        전체 학습 실행 (백그라운드)"
        echo "  prepare      데이터 준비만 실행"
        echo "  yolo-train [epochs]  YOLO 학습만 실행"
        echo "  eval         모델 평가"
        echo "  demo <img>   단일 이미지 파이프라인 데모"
        echo "  logs         로그 실시간 확인"
        echo "  status       컨테이너 및 GPU 상태"
        echo ""
        echo "일반적인 사용 순서:"
        echo "  1. ./run_docker.sh build   # 이미지 빌드 (최초)"
        echo "  2. ./run_docker.sh up      # 컨테이너 시작"
        echo "  3. ./run_docker.sh train   # 학습 시작 (SSH 종료해도 계속 실행)"
        echo "  4. ./run_docker.sh logs    # 진행상황 확인"
        echo "  5. ./run_docker.sh demo image.jpg  # 결과 확인"
        ;;

    *)
        print_error "알 수 없는 명령: $1"
        echo "  사용법: ./run_docker.sh help"
        exit 1
        ;;
esac
