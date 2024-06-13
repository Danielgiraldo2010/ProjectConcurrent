# Proyecto Final:

## Análisis de Rendimiento de Dotplot

## Secuencial vs Paralelización

El objetivo de este proyecto es implementar y analizar el rendimiento de tres formas de realizar un dotplot, una técnica comúnmente utilizada en bioinformática para comparar secuencias de ADN o proteínas.

### Prerequisitos

Este proyecto fue desarrollado utilizando Python 3.11.7 y aprovecha la computación paralela a través de las librerías multiprocessing y mpi4py. Requiere como entrada una secuencia de referencia y una de consulta en formato .fna, las cuales deben ser especificadas en la línea de comandos para calcular el dot-plot.

### Instalaciones

Primero, asegúrese de tener Python instalado. Luego, instale las librerías necesarias:

```
pip install numpy
pip install mpi4py
pip install biopython
pip install matplotlib
pip install tqdm==2.2.3
pip install opencv-python

```

Por ultimo, clone el repositorio del proyecto:

```
git clone https://github.com/Danielgiraldo2010/ProjectConcurrent.git
          
```

### Ejecución

Para ejecutar el programa en modo secuencial, use el siguiente comando:

```
python ProjectConcurrent.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --sequential

```

Para ejecutar el programa utilizando multiprocessing, use este comando:

```
python ProjectConcurrent.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --multiprocessing
```

Para ejecutar el programa con mpi4py, utilice el siguiente comando:

```
python ProjectConcurrent.py --num_processes 1 2 3 4 --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --mpi
```
