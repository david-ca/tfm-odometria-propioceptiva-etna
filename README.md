# Estimación de la posición de un róver con sensores propioceptivos y aprendizaje profundo

Código del Trabajo Fin de Máster «Estimación de la posición en robots móviles terrestres utilizando sensores propioceptivos y herramientas de IA» (Máster Universitario en Inteligencia Artificial, Universidad Internacional de Valencia, 2026).

Autor: David Carrasco Alonso · Director: Jesús Marcey García

## Qué hace

Se entrenan modelos de aprendizaje profundo (GRU, TCN y Transformer) que estiman, a partir de la IMU y de la odometría de ruedas, el desplazamiento de un róver en cada intervalo de un segundo (avance, deslizamiento lateral y componente vertical). El resultado se compara con la odometría de ruedas, sin calibrar y calibrada, y se reconstruye la trayectoria por tramos.

## Datos

Se emplean los recorridos del róver LRU (DLR) en el monte Etna, publicados por el proyecto ROBEX:

- Vayugundla, M., Steidle, F., Smisek, M., Schuster, M. J., Bussmann, K., y Wedler, A. (2018). Datasets of long range navigation experiments in a moon analogue environment on Mount Etna. ISR 2018.
- Descarga: https://www.robex-allianz.de/datasets/

Los datos no se incluyen en este repositorio. Para reproducir el trabajo, descárgalos en una carpeta local.

**Rutas.** Los scripts definen al principio las carpetas de trabajo con rutas absolutas (variables como `BASE` y `RES`, por ejemplo `C:\Users\Usuario\Desktop\TFM\results`). Antes de ejecutarlos, cámbialas por las rutas de tu equipo.

## Estructura

| Carpeta | Contenido |
|---|---|
| raíz | Scripts del flujo principal, en el orden de ejecución que se indica abajo |
| `analysis/` | Scripts que generan las figuras y los estadísticos de la memoria |
| `exploration/` | Scripts de diagnóstico y versiones anteriores de la evaluación (partición entre recorridos) |
| `results/` | Resultados de los experimentos (tablas CSV, ficheros JSON y figuras) |

La relación entre cada resultado de la memoria, el script que lo produce y el fichero en que se guarda está en la Tabla 10 (Apéndice I.5 de la memoria).

## Entorno

Python 3.11 con las bibliotecas de `requirements.txt`. Los entrenamientos se ejecutaron con PyTorch 2.13 y CUDA 12.6 en una GPU NVIDIA GeForce RTX 3060.

```
pip install -r requirements.txt
```

## Orden de ejecución

```
python extract_topics.py
python build_target.py
python build_dataset.py
python build_dataset_mixed.py
python run_experiments.py --stage all --search_seeds 3
python baseline_mixed_paired.py
python calibrated_baseline.py
python reconstruct_mixed.py
python analysis/make_figures.py results figuras
python analysis/analyze_selected_model.py
```

Las particiones se generan con una semilla fija (42) y cada modelo se entrena con las semillas 0 a 4. Algunas operaciones en GPU no son deterministas, por lo que una nueva ejecución puede dar diferencias pequeñas, del orden de la variación entre inicializaciones.
