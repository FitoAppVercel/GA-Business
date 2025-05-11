"""
Módulo que contiene funciones auxiliares para el algoritmo genético.
Incluye funciones para la generación dummy de ideas y simulación de evaluación de fitness.
"""

import random
from models import Idea

# Lista de posibles problemas para generar ideas dummy
SAMPLE_PROBLEMS = [
    "La dificultad para encontrar productos locales de calidad",
    "El exceso de plástico en los empaques de alimentos",
    "La falta de tiempo para cocinar comidas saludables",
    "La dificultad para transportarse en ciudades congestionadas",
    "El alto costo de la vivienda en zonas urbanas",
    "La dificultad para reciclar correctamente",
    "La pérdida de tiempo en tareas administrativas repetitivas",
    "La falta de espacios verdes en zonas urbanas",
    "El acceso limitado a servicios médicos especializados",
    "La dificultad para encontrar trabajo remoto de calidad"
]

# Lista de posibles soluciones para generar ideas dummy
SAMPLE_SOLUTIONS = [
    "Una plataforma que conecta consumidores con agricultores locales",
    "Empaques biodegradables hechos de materiales orgánicos",
    "Un servicio de preparación y entrega de comidas saludables personalizadas",
    "Una aplicación de micromovilidad compartida",
    "Un modelo de co-living con espacios compartidos",
    "Un sistema de clasificación automática de residuos mediante IA",
    "Un asistente virtual basado en IA para automatizar tareas administrativas",
    "Jardines verticales modulares para espacios urbanos",
    "Una plataforma de telemedicina especializada",
    "Un marketplace de trabajos remotos verificados"
]

def generate_random_idea():
    """
    Genera una idea aleatoria combinando un problema y una solución de las listas predefinidas.
    
    Returns:
        Idea: Una nueva instancia de Idea con un problema y solución aleatorios.
    """
    problem = random.choice(SAMPLE_PROBLEMS)
    solution = random.choice(SAMPLE_SOLUTIONS)
    return Idea(problem=problem, solution=solution)

def generate_initial_population(size):
    """
    Genera una población inicial de ideas aleatorias.
    
    Args:
        size (int): Tamaño de la población a generar.
        
    Returns:
        list: Lista de instancias de Idea generadas aleatoriamente.
    """
    return [generate_random_idea() for _ in range(size)]

def simulate_llm_evaluation(idea):
    """
    Simula la evaluación de una idea por un modelo de lenguaje.
    Esta es una función dummy que asigna una puntuación aleatoria ponderada.
    
    Args:
        idea (Idea): La idea a evaluar.
        
    Returns:
        float: Puntuación de aptitud simulada entre 0.0 y 1.0.
    """
    # Simulamos coherencia entre problema y solución
    # En un caso real, esto sería evaluado por un LLM
    problem_words = set(idea.problem.lower().split())
    solution_words = set(idea.solution.lower().split())
    
    # Simulamos una métrica de relevancia basada en palabras comunes
    common_words = len(problem_words.intersection(solution_words))
    problem_relevance = min(1.0, common_words / 3.0) if common_words > 0 else 0.1
    
    # Añadimos un componente aleatorio para simular evaluación subjetiva
    innovation_score = random.uniform(0.3, 1.0)
    feasibility_score = random.uniform(0.2, 0.9)
    market_potential = random.uniform(0.1, 1.0)
    
    # Calculamos el fitness combinando las métricas con pesos
    fitness = (
        0.3 * problem_relevance +
        0.3 * innovation_score +
        0.2 * feasibility_score +
        0.2 * market_potential
    )
    
    return fitness

def simulate_llm_mutation(idea, mutation_strength=0.5):
    """
    Simula la mutación de una idea por un modelo de lenguaje.
    Esta es una función dummy que modifica ligeramente el problema o la solución.
    
    Args:
        idea (Idea): La idea a mutar.
        mutation_strength (float): Intensidad de la mutación entre 0.0 y 1.0.
        
    Returns:
        Idea: Una nueva idea mutada.
    """
    new_idea = idea.copy()
    
    # Decidimos si mutamos el problema, la solución o ambos
    if random.random() < 0.5:
        # Mutar el problema
        if random.random() < mutation_strength:
            new_idea.problem = random.choice(SAMPLE_PROBLEMS)
        else:
            # Simulamos pequeños cambios
            words = new_idea.problem.split()
            if len(words) > 3:
                random_index = random.randint(0, len(words) - 1)
                words[random_index] = random.choice(["mayor", "menor", "creciente", "frecuente", "común"])
                new_idea.problem = " ".join(words)
    else:
        # Mutar la solución
        if random.random() < mutation_strength:
            new_idea.solution = random.choice(SAMPLE_SOLUTIONS)
        else:
            # Simulamos pequeños cambios
            words = new_idea.solution.split()
            if len(words) > 3:
                random_index = random.randint(0, len(words) - 1)
                words[random_index] = random.choice(["innovador", "eficiente", "sostenible", "inteligente", "accesible"])
                new_idea.solution = " ".join(words)
    
    return new_idea

def simulate_llm_crossover(idea1, idea2):
    """
    Simula la recombinación de dos ideas por un modelo de lenguaje.
    Esta es una función dummy que combina elementos de dos ideas.
    
    Args:
        idea1 (Idea): Primera idea para la recombinación.
        idea2 (Idea): Segunda idea para la recombinación.
        
    Returns:
        tuple: Un par de nuevas ideas resultantes del cruce.
    """
    # Primera idea hijo: problema de idea1, solución de idea2
    child1 = Idea(problem=idea1.problem, solution=idea2.solution)
    
    # Segunda idea hijo: problema de idea2, solución de idea1
    child2 = Idea(problem=idea2.problem, solution=idea1.solution)
    
    return child1, child2