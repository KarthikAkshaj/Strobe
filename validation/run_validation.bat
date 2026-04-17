@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Phase-2B Validation Runner
REM Processes all test videos and collects diagnostics output.
REM ============================================================

set PROJECT_ROOT=%~dp0..
set PYTHON_TOOLS=%PROJECT_ROOT%\python_tools
set TEST_VIDEOS=%~dp0test_videos
set OUTPUTS=%~dp0outputs

REM Create timestamped run directory
for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do (
    set TIMESTAMP=%%c%%a%%b_%%d%%e%%f
)
set RUN_DIR=%OUTPUTS%\run_%TIMESTAMP%

echo ============================================================
echo Phase-2B Validation Run
echo Run ID: run_%TIMESTAMP%
echo ============================================================
echo.

REM Create output directories
for %%C in (static single_scene_dynamic multi_scene gradual_transition noisy adversarial) do (
    mkdir "%RUN_DIR%\%%C" 2>nul
)
mkdir "%RUN_DIR%\reviews" 2>nul

set TOTAL=0
set SUCCESS=0
set FAILED=0

REM Process each category
for %%C in (static single_scene_dynamic multi_scene gradual_transition noisy adversarial) do (
    echo.
    echo === Category: %%C ===
    echo.

    for %%V in ("%TEST_VIDEOS%\%%C\*.mp4") do (
        set /a TOTAL+=1
        set VIDNAME=%%~nV

        echo Processing: %%~nxV
        echo   Category: %%C
        echo   Output:   %RUN_DIR%\%%C\!VIDNAME!.captions.json

        python "%PYTHON_TOOLS%\process_video.py" "%%V" --output "%RUN_DIR%\%%C\!VIDNAME!.captions.json"

        if !errorlevel! equ 0 (
            set /a SUCCESS+=1
            echo   Status: OK
        ) else (
            set /a FAILED+=1
            echo   Status: FAILED
        )
        echo.
    )
)

REM Copy review template for each output
echo.
echo Creating review files...
for %%C in (static single_scene_dynamic multi_scene gradual_transition noisy adversarial) do (
    for %%V in ("%RUN_DIR%\%%C\*.captions.json") do (
        set VIDNAME=%%~nV
        set REVIEWNAME=!VIDNAME:.captions=!

        (
            echo ================================================================================
            echo PHASE-2B VIDEO REVIEW
            echo ================================================================================
            echo Video: !REVIEWNAME!
            echo Category: %%C
            echo Date Reviewed:
            echo Reviewer:
            echo.
            echo --------------------------------------------------------------------------------
            echo SUMMARY METRICS ^(copy from diagnostics.summary in JSON^)
            echo --------------------------------------------------------------------------------
            echo Total Events:
            echo Events per Minute:
            echo.
            echo Confidence:
            echo   Mean:
            echo   Std:
            echo.
            echo Confidence Gap:
            echo   Mean:
            echo.
            echo Change Similarity:
            echo   Mean:
            echo.
            echo Stability:
            echo   Suppression Rate:
            echo   Max Consecutive:
            echo.
            echo Captions:
            echo   Unique Count:
            echo   Entropy:
            echo.
            echo --------------------------------------------------------------------------------
            echo VIDEO-LEVEL ANOMALY FLAGS
            echo --------------------------------------------------------------------------------
            echo Flags:
            echo.
            echo Event Flag Counts:
            echo.
            echo --------------------------------------------------------------------------------
            echo SPOT CHECK ^(2-3 events manually verified^)
            echo --------------------------------------------------------------------------------
            echo Event ___ @ ___s:
            echo   Caption:
            echo   Confidence:
            echo   Visual Match: [CORRECT / PARTIAL / WRONG]
            echo   Notes:
            echo.
            echo Event ___ @ ___s:
            echo   Caption:
            echo   Confidence:
            echo   Visual Match: [CORRECT / PARTIAL / WRONG]
            echo   Notes:
            echo.
            echo --------------------------------------------------------------------------------
            echo REVIEWER NOTES
            echo --------------------------------------------------------------------------------
            echo.
            echo.
            echo --------------------------------------------------------------------------------
            echo VERDICT
            echo --------------------------------------------------------------------------------
            echo [ ] PASS       - Metrics healthy, spot checks pass, no blocking flags
            echo [ ] BORDERLINE - Minor concerns, acceptable for this category
            echo [ ] FAIL       - Blocking issues found
            echo.
            echo Blocking Issue ^(if FAIL^):
            echo.
            echo ================================================================================
        ) > "%RUN_DIR%\reviews\!REVIEWNAME!.review.txt"
    )
)

echo.
echo ============================================================
echo Validation Run Complete
echo ============================================================
echo Run directory: %RUN_DIR%
echo Total videos:  %TOTAL%
echo Succeeded:     %SUCCESS%
echo Failed:        %FAILED%
echo.
echo Next steps:
echo   1. Review output JSONs in %RUN_DIR%\
echo   2. Fill in review files in %RUN_DIR%\reviews\
echo   3. Run: python validation\aggregate_results.py "%RUN_DIR%"
echo   4. Complete exit_report.md
echo ============================================================

endlocal
