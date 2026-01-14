# Optimizador de Corte de Varillas 🔧

Sistema profesional para optimizar el corte de varillas de construcción, minimizando desperdicios y costos mediante reutilización inteligente de sobrantes.

## Características

✅ **Optimización inteligente**: Encuentra la mejor combinación de varillas estándar
✅ **Reutilización de sobrantes**: Aprovecha desperdicios para piezas posteriores (ahorra hasta 79+ piezas)
✅ **Minimiza desperdicios**: Calcula el menor desperdicio posible
✅ **Trazabilidad completa**: IDs únicos muestran qué pedido generó cada sobrante reutilizado
✅ **Reportes detallados**: Genera plan de corte, lista de compra y orden para proveedor
✅ **Exportación a Excel**: Crea archivos profesionales con formato
✅ **Múltiples diámetros**: Soporta todos los diámetros estándar (3/8" a 1")
✅ **Ejecutable standalone**: No requiere Python instalado (.exe para Windows)

## Referencias Estándar del Mercado

| Diámetro | Referencia 1 | Referencia 2 | Referencia 3 |
|----------|--------------|--------------|--------------|
| 3/8"     | 6m           | 9m           | 12m          |
| 1/2"     | 6m           | 9m           | 12m          |
| 5/8"     | 6m           | 9m           | 12m          |
| 3/4"     | 6m           | 9m           | 12m          |
| 7/8"     | 6m           | 9m           | 12m          |
| 1"       | 6m           | 9m           | 12m          |

## Instalación y Uso

### Opción 1: Ejecutable (Recomendado - No requiere Python)

1. **Descarga el ejecutable**:
   - Ve a la carpeta `dist/` del repositorio
   - Descarga `OptimizadorCortes.exe`, `Ejecutar.bat` e `INSTRUCCIONES.txt`

2. **Usa el programa**:
   - Coloca tu archivo Excel (por ejemplo `Cortes.xlsx`) en la misma carpeta
   - **Forma fácil**: Haz doble clic en `Ejecutar.bat`
   - **Arrastrando**: Arrastra tu archivo .xlsx sobre `OptimizadorCortes.exe`
   - **Línea de comandos**: `OptimizadorCortes.exe MiArchivo.xlsx`

3. **Resultados**:
   - Se generarán 2 archivos Excel:
     - `[Nombre]_PLAN_CORTE_OPTIMIZADO.xlsx` - Plan detallado con 3 hojas
     - `[Nombre]_ORDEN_COMPRA.xlsx` - Orden consolidada para proveedor

### Opción 2: Desde código Python

1. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

2. **Ejecutar**:
```bash
python optimizador_varillas.py
```

3. El programa buscará `Cortes.xlsx` en la carpeta `Downloads`

## Formato del Excel de Entrada

El archivo debe contener las columnas:
- `Element Qty`: Cantidad de elementos
- `Ø (in)`: Diámetro en pulgadas
- `Length (m)`: Longitud requerida en metros

## Salidas

### 1. Plan de Corte Optimizado (3 hojas):
- **Plan de Corte**: Detalle completo de cada pedido con IDs únicos
- **Lista de Compra**: Resumen consolidado por diámetro y referencia
- **Trazabilidad Sobrantes**: Seguimiento de qué sobrante se usó dónde

### 2. Orden de Compra (2 hojas):
- **Orden de Compra**: Tabla profesional para enviar al proveedor
- **Resumen por Diámetro**: Totales agrupados

### 3. Consola:
- Muestra en tiempo real el proceso de optimización
- Indica cuántas piezas se obtuvieron de sobrantes
- Desperdicio total calculado
2. **Archivo Excel**: `Cortes_PLAN_CORTE.xlsx` con detalles completos
   - Diámetro
   - Longitud de piezas
   - Cantidad de piezas
   - Varillas a usar
   - Plan de corte
   - Desperdicio calculado
   - Eficiencia porcentual

## Algoritmo de Optimización

El sistema evalúa todas las combinaciones posibles de varillas estándar y selecciona la que produce:
1. Menor desperdicio total
2. Menor cantidad de varillas
3. Máxima eficiencia de corte

---
Desarrollado para optimizar proyectos de construcción 🏗️
