@echo off
chcp 65001 >nul
cls
echo ========================================
echo  OPTIMIZADOR DE CORTE DE VARILLAS
echo ========================================
echo.

REM Verificar si se pasó un archivo como argumento
if "%~1"=="" (
    REM No hay argumento, buscar Cortes.xlsx
    if exist "Cortes.xlsx" (
        echo ✓ Archivo encontrado: Cortes.xlsx
        echo.
        echo Procesando...
        echo.
        OptimizadorCortes.exe "Cortes.xlsx"
    ) else (
        echo ✗ No se encontró el archivo "Cortes.xlsx"
        echo.
        echo Por favor:
        echo 1. Coloca tu archivo Excel en esta carpeta y renómbralo a "Cortes.xlsx"
        echo 2. O arrastra tu archivo Excel sobre este .bat
        echo.
        pause
        exit /b 1
    )
) else (
    REM Hay argumento, usar ese archivo
    if exist "%~1" (
        echo ✓ Archivo encontrado: %~nx1
        echo.
        echo Procesando...
        echo.
        OptimizadorCortes.exe "%~1"
    ) else (
        echo ✗ El archivo "%~1" no existe
        echo.
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo ✓ PROCESO COMPLETADO
echo ========================================
echo.
echo Archivos generados:
echo - Plan de corte optimizado
echo - Orden de compra
echo.
pause
