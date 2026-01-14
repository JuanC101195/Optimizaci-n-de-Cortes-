@echo off
chcp 65001 >nul
cls
echo ========================================
echo  OPTIMIZADOR DE CORTE DE VARILLAS
echo ========================================
echo.
echo Iniciando optimizador...
echo.

REM Ejecutar el optimizador (pedirá el archivo al usuario)
OptimizadorCortes.exe

echo.
echo ========================================
pause
