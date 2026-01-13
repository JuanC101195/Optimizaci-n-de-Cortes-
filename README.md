# Optimizador de Corte de Varillas 🔧

Sistema profesional para optimizar el corte de varillas de construcción, minimizando desperdicios y costos.

## Características

✅ **Optimización inteligente**: Encuentra la mejor combinación de varillas estándar
✅ **Minimiza desperdicios**: Calcula el menor desperdicio posible
✅ **Reportes detallados**: Genera plan de corte y lista de compra
✅ **Exportación a Excel**: Crea archivo con el plan optimizado
✅ **Múltiples diámetros**: Soporta todos los diámetros estándar (3/8" a 1")

## Referencias Estándar del Mercado

| Diámetro | Referencia 1 | Referencia 2 | Referencia 3 |
|----------|--------------|--------------|--------------|
| 3/8"     | 6m           | 9m           | 12m          |
| 1/2"     | 6m           | 9m           | 12m          |
| 5/8"     | 6m           | 9m           | 12m          |
| 3/4"     | 6m           | 9m           | 12m          |
| 7/8"     | 6m           | 9m           | 12m          |
| 1"       | 6m           | 9m           | 12m          |

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

1. Coloca tu archivo Excel en la carpeta `Downloads` con el nombre `Cortes.xlsx`
2. Ejecuta el optimizador:

```bash
python optimizador_varillas.py
```

3. Revisa el reporte en consola y el archivo generado `Cortes_PLAN_CORTE.xlsx`

## Formato del Excel de Entrada

El archivo debe contener las columnas:
- `Element Qty`: Cantidad de elementos
- `Ø (in)`: Diámetro en pulgadas
- `Length (m)`: Longitud requerida en metros

## Salidas

1. **Reporte en consola**: Muestra el plan de corte optimizado
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
