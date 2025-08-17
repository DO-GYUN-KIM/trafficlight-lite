@echo off
REM ===== YOLO-lite 콘다 환경 활성화 =====
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate yolo-lite

REM ===== 프로젝트 폴더로 이동 =====
cd /d C:\summer\trafficlight-lite\tools

REM ===== GUI 실행 =====
python gui_trafficlight_system.py

REM ===== 창 닫힘 방지 =====
echo.
echo (종료하려면 아무 키나 누르세요)
pause >nul