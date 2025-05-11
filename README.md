# GA-Business: Algoritmo Genético para Generación de Ideas de Negocio

## Descripción General

GA-Business es un sistema que utiliza algoritmos genéticos para generar, evaluar y evolucionar ideas de negocio de forma automática. Cada "individuo" en la población del algoritmo genético representa una idea de negocio compuesta por dos elementos principales:

1. **Problema**: Una oración que describe un problema o necesidad del mercado
2. **Solución**: Una oración que propone una solución al problema identificado

El sistema aplica principios de evolución artificial para mejorar iterativamente la calidad de las ideas a través de múltiples generaciones, utilizando operaciones de selección, recombinación y mutación.

Actualmente, el comportamiento del sistema se simula mediante funciones "dummy" que imitan la generación y evaluación de ideas, pero la arquitectura está diseñada para integrar fácilmente modelos de lenguaje (LLMs) como GPT-4 o Claude en futuras versiones.

## Estructura del Código

El proyecto está organizado en los siguientes archivos:

- **models.py**: Define la clase `Idea` que representa el cromosoma genético (problema + solución + fitness).
- **utils.py**: Contiene funciones auxiliares para generar ideas aleatorias y simular evaluación, mutación y recombinación.
- **genetic.py**: Implementa las funciones principales del algoritmo genético (inicialización, evaluación, selección, recombinación, mutación y evolución).
- **main.py**: Script principal para ejecutar el algoritmo con diferentes configuraciones y visualizar resultados.

## Requisitos

- Python 3.7 o superior
- Matplotlib (para visualizar la evolución del fitness)

## Instrucciones de Uso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/GA-Business.git
cd GA-Business
```

### 2. Configurar Entorno Virtual e Instalar Dependencias

Crear y activar un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

Instalar dependencias:

```bash
pip3 install -r requirements.txt
```

O instalar sólo matplotlib:

```bash
pip3 install matplotlib
```

### 3. Ejecutar el Algoritmo

Ejecución básica:

```bash
python3 main.py
```

Ejecución con parámetros personalizados:

```bash
python3 main.py --population 30 --generations 20 --mutation-rate 0.3 --crossover-rate 0.7
```

Parámetros disponibles:

- `--population`: Tamaño de la población (default: 20)
- `--generations`: Número de generaciones (default: 10)
- `--elite`: Número de individuos elite a preservar (default: 2)
- `--crossover-rate`: Tasa de cruce (default: 0.8)
- `--mutation-rate`: Tasa de mutación (default: 0.2)
- `--mutation-strength`: Intensidad de la mutación (default: 0.5)
- `--seed`: Semilla para generación de números aleatorios (opcional)
- `--no-plot`: No mostrar gráfico de evolución

## Ejemplo de Salida Esperada

Al ejecutar el algoritmo, obtendrás resultados similares a estos:

```
=== Generación 10 ===
Fitness promedio: 0.6243
Mejor fitness: 0.8762
Peor fitness: 0.4328

Mejores ideas:

1. Fitness: 0.8762
   Problema: La dificultad para encontrar productos locales de calidad
   Solución: Una plataforma que conecta consumidores con agricultores locales

2. Fitness: 0.8215
   Problema: La pérdida de tiempo en tareas administrativas repetitivas
   Solución: Un asistente virtual basado en IA para automatizar tareas administrativas

3. Fitness: 0.7854
   Problema: La falta de espacios verdes en zonas urbanas
   Solución: Jardines verticales modulares para espacios urbanos
```

Además, se generará un gráfico `fitness_evolution.png` que muestra la evolución del fitness promedio a lo largo de las generaciones.

## Notas Futuras

Para futuras versiones, se planea:

1. Integrar modelos de lenguaje (LLMs) reales como GPT-4 o Claude Sonnet para:
   - Generar ideas iniciales más sofisticadas
   - Evaluar ideas con criterios como originalidad, viabilidad, escalabilidad y potencial de mercado
   - Realizar operaciones de recombinación y mutación que preserven la coherencia semántica

2. Añadir más detalles a cada idea, como:
   - Segmento de mercado
   - Propuesta de valor
   - Modelo de monetización
   - Estrategia de crecimiento

3. Implementar visualización interactiva de la evolución de las ideas

4. Explorar técnicas de nicho para mantener la diversidad en la población

---

© 2025 GA-Business