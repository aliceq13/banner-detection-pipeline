#!/bin/bash
# ============================================================
# run_training.sh - SSH 연결이 끊겨도 계속 실행되는 학습 스크립트
# ============================================================
#
# SSH 세션이 끊기면 프로세스가 죽는 문제를 해결하는 방법:
#   1. nohup: 터미널 종료 시 HUP 시그널 무시
#   2. &:     백그라운드 실행
#   3. tee:   화면과 로그 파일에 동시 출력
#
# 사용법:
#   chmod +x run_training.sh
#   ./run_training.sh              # 전체 파이프라인 실행
#   ./run_training.sh --skip-data  # 데이터 준비 건너뜀
#   ./run_training.sh --epochs 50  # 에폭 수 변경
#
# 실행 후 SSH 끊기:
#   - 로그 확인: tail -f logs/training_YYYYMMDD_HHMMSS.log
#   - 상태 확인: cat logs/training.pid | xargs ps -p
#   - 중단:     cat logs/training.pid | xargs kill
# ============================================================

# 스크립트 오류 시 즉시 종료
set -e

# ── 설정 변수 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/training_${TIMESTAMP}.log"
PID_FILE="${LOGS_DIR}/training.pid"
SKIP_DATA=false
EXTRA_ARGS=""

# ── 인자 파싱 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --epochs)
            EXTRA_ARGS="$EXTRA_ARGS --epochs $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# ── 디렉토리 생성 ───────────────────────────────────────────
mkdir -p "${LOGS_DIR}"

# ── Docker가 실행 중인지 확인 ───────────────────────────────
if command -v docker &> /dev/null && docker ps | grep -q "banner_project"; then
    echo "Docker 컨테이너에서 실행합니다..."
    RUN_PREFIX="docker exec banner_project python /workspace/project/"
    CHECK_CMD="docker exec banner_project"
else
    echo "호스트에서 직접 실행합니다..."
    RUN_PREFIX="python ${SCRIPT_DIR}/"
    CHECK_CMD=""
fi

# ── 실제 학습 함수 ─────────────────────────────────────────
run_pipeline() {
    echo "========================================"
    echo "현수막 탐지 파이프라인 학습 시작"
    echo "시작 시각: $(date)"
    echo "로그 파일: ${LOG_FILE}"
    echo "========================================"

    # Step 1: 데이터 준비
    if [ "$SKIP_DATA" = false ]; then
        echo ""
        echo "[Step 1/3] 데이터 준비 (01_prepare_data.py)"
        echo "----------------------------------------"
        if [ -n "$CHECK_CMD" ]; then
            docker exec banner_project python /workspace/project/01_prepare_data.py
        else
            python "${SCRIPT_DIR}/01_prepare_data.py"
        fi
    else
        echo "[Step 1/3] 데이터 준비 건너뜀 (--skip-data)"
    fi

    # Step 2: YOLO 학습
    echo ""
    echo "[Step 2/3] YOLO26 학습 (02_train_yolo.py)"
    echo "----------------------------------------"
    if [ -n "$CHECK_CMD" ]; then
        docker exec banner_project python /workspace/project/02_train_yolo.py $EXTRA_ARGS
    else
        python "${SCRIPT_DIR}/02_train_yolo.py" $EXTRA_ARGS
    fi

    # Step 3: 평가
    echo ""
    echo "[Step 3/3] 모델 평가 (03_evaluate_yolo.py)"
    echo "----------------------------------------"
    if [ -n "$CHECK_CMD" ]; then
        docker exec banner_project python /workspace/project/03_evaluate_yolo.py
    else
        python "${SCRIPT_DIR}/03_evaluate_yolo.py"
    fi

    echo ""
    echo "========================================"
    echo "학습 완료: $(date)"
    echo "========================================"
}

# ── 백그라운드 실행 (nohup + tee로 로그 저장) ──────────────
echo "백그라운드 실행 시작..."
echo "로그: ${LOG_FILE}"
echo ""

# nohup: SSH 종료 후에도 실행 유지
# tee:   화면에 출력하면서 로그 파일에도 저장
nohup bash -c "$(declare -f run_pipeline); run_pipeline" 2>&1 | tee "${LOG_FILE}" &

# PID 저장 (나중에 상태 확인/중단에 사용)
echo $! > "${PID_FILE}"
PID=$!

echo "========================================"
echo "프로세스 시작됨 (PID: ${PID})"
echo ""
echo "로그 실시간 확인:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "실행 상태 확인:"
echo "  ps -p ${PID}"
echo ""
echo "학습 중단:"
echo "  kill ${PID}"
echo "========================================"
