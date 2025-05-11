"""
Módulo que contiene las funciones principales del algoritmo genético para
la generación y evaluación de ideas de negocio.
"""

import random
from typing import List, Tuple, Callable
from models import Idea
from utils import simulate_llm_evaluation, simulate_llm_mutation, simulate_llm_crossover

def initialize_population(population_size: int, initializer_func: Callable[[], Idea]) -> List[Idea]:
    """
    Inicializa una población de ideas.
    
    Args:
        population_size (int): Tamaño de la población a generar.
        initializer_func (callable): Función que genera una instancia de Idea.
        
    Returns:
        List[Idea]: Población inicial de ideas.
    """
    return [initializer_func() for _ in range(population_size)]

def evaluate_population(population: List[Idea], evaluator_func: Callable[[Idea], float]) -> List[Idea]:
    """
    Evalúa la aptitud de cada idea en la población.
    
    Args:
        population (List[Idea]): Lista de ideas a evaluar.
        evaluator_func (callable): Función que evalúa una idea y devuelve un valor de aptitud.
        
    Returns:
        List[Idea]: Lista de ideas con sus valores de aptitud actualizados.
    """
    for idea in population:
        idea.fitness = evaluator_func(idea)
    return population

def select_parents(population: List[Idea], num_parents: int, tournament_size: int = 3) -> List[Idea]:
    """
    Selecciona padres de la población utilizando selección por torneo.
    
    Args:
        population (List[Idea]): Lista de ideas evaluadas.
        num_parents (int): Número de padres a seleccionar.
        tournament_size (int): Tamaño del torneo para cada selección.
        
    Returns:
        List[Idea]: Lista de ideas seleccionadas como padres.
    """
    parents = []
    for _ in range(num_parents):
        # Seleccionamos aleatoriamente individuos para el torneo
        tournament = random.sample(population, min(tournament_size, len(population)))
        # Elegimos el individuo con mayor aptitud del torneo
        winner = max(tournament, key=lambda idea: idea.fitness)
        parents.append(winner)
    return parents

def crossover(parents: List[Idea], crossover_func: Callable[[Idea, Idea], Tuple[Idea, Idea]], 
              crossover_rate: float = 0.8) -> List[Idea]:
    """
    Realiza el cruce entre padres para generar nuevos hijos.
    
    Args:
        parents (List[Idea]): Lista de ideas seleccionadas como padres.
        crossover_func (callable): Función que realiza el cruce entre dos ideas.
        crossover_rate (float): Probabilidad de que ocurra el cruce.
        
    Returns:
        List[Idea]: Lista de ideas hijo generadas.
    """
    offspring = []
    
    # Asegurarnos de tener un número par de padres
    if len(parents) % 2 != 0:
        parents.append(random.choice(parents))
    
    # Emparejar padres y realizar cruce
    random.shuffle(parents)
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1]
        
        if random.random() < crossover_rate:
            child1, child2 = crossover_func(parent1, parent2)
        else:
            # Si no hay cruce, los hijos son copias de los padres
            child1, child2 = parent1.copy(), parent2.copy()
            
        offspring.extend([child1, child2])
    
    return offspring

def mutate(population: List[Idea], mutation_func: Callable[[Idea, float], Idea], 
           mutation_rate: float = 0.2, mutation_strength: float = 0.5) -> List[Idea]:
    """
    Aplica mutación a una población de ideas.
    
    Args:
        population (List[Idea]): Lista de ideas a mutar.
        mutation_func (callable): Función que realiza la mutación de una idea.
        mutation_rate (float): Probabilidad de que ocurra la mutación.
        mutation_strength (float): Intensidad de la mutación cuando ocurre.
        
    Returns:
        List[Idea]: Lista de ideas mutadas.
    """
    mutated_population = []
    
    for idea in population:
        if random.random() < mutation_rate:
            mutated_idea = mutation_func(idea, mutation_strength)
            mutated_population.append(mutated_idea)
        else:
            mutated_population.append(idea)
    
    return mutated_population

def elitism(current_population: List[Idea], offspring: List[Idea], 
            elite_size: int) -> List[Idea]:
    """
    Implementa elitismo preservando los mejores individuos de la generación actual.
    
    Args:
        current_population (List[Idea]): Población actual.
        offspring (List[Idea]): Nueva descendencia generada.
        elite_size (int): Número de elite a preservar.
        
    Returns:
        List[Idea]: Nueva población con elite preservada.
    """
    # Ordenamos la población actual por fitness (de mayor a menor)
    sorted_population = sorted(current_population, key=lambda idea: idea.fitness, reverse=True)
    
    # Seleccionamos la elite
    elite = sorted_population[:elite_size]
    
    # Completamos la nueva generación
    new_population = elite + offspring[:len(current_population) - elite_size]
    
    return new_population

def evolve(population: List[Idea], 
           evaluator_func: Callable[[Idea], float],
           crossover_func: Callable[[Idea, Idea], Tuple[Idea, Idea]],
           mutation_func: Callable[[Idea, float], Idea],
           crossover_rate: float = 0.8,
           mutation_rate: float = 0.2,
           mutation_strength: float = 0.5,
           elite_size: int = 2,
           tournament_size: int = 3) -> List[Idea]:
    """
    Evoluciona una población de ideas a través de una generación.
    
    Args:
        population (List[Idea]): Población actual de ideas.
        evaluator_func (callable): Función que evalúa la aptitud de una idea.
        crossover_func (callable): Función para realizar el cruce entre ideas.
        mutation_func (callable): Función para mutar ideas.
        crossover_rate (float): Tasa de cruce.
        mutation_rate (float): Tasa de mutación.
        mutation_strength (float): Intensidad de la mutación.
        elite_size (int): Número de individuos elite a preservar.
        tournament_size (int): Tamaño del torneo para la selección.
        
    Returns:
        List[Idea]: Nueva población evolucionada.
    """
    # Evaluar la población actual
    evaluate_population(population, evaluator_func)
    
    # Seleccionar padres
    num_parents = len(population)
    parents = select_parents(population, num_parents, tournament_size)
    
    # Generar hijos mediante cruce
    offspring = crossover(parents, crossover_func, crossover_rate)
    
    # Aplicar mutación a los hijos
    mutated_offspring = mutate(offspring, mutation_func, mutation_rate, mutation_strength)
    
    # Implementar elitismo
    new_population = elitism(population, mutated_offspring, elite_size)
    
    # Evaluar la nueva población
    evaluate_population(new_population, evaluator_func)
    
    return new_population

def run_genetic_algorithm(initializer_func: Callable[[], Idea],
                         evaluator_func: Callable[[Idea], float],
                         crossover_func: Callable[[Idea, Idea], Tuple[Idea, Idea]],
                         mutation_func: Callable[[Idea, float], Idea],
                         population_size: int,
                         num_generations: int,
                         crossover_rate: float = 0.8,
                         mutation_rate: float = 0.2,
                         mutation_strength: float = 0.5,
                         elite_size: int = 2,
                         tournament_size: int = 3) -> Tuple[List[Idea], List[float]]:
    """
    Ejecuta el algoritmo genético completo durante múltiples generaciones.
    
    Args:
        initializer_func (callable): Función para inicializar ideas.
        evaluator_func (callable): Función para evaluar ideas.
        crossover_func (callable): Función para realizar cruce.
        mutation_func (callable): Función para realizar mutación.
        population_size (int): Tamaño de la población.
        num_generations (int): Número de generaciones a evolucionar.
        crossover_rate (float): Tasa de cruce.
        mutation_rate (float): Tasa de mutación.
        mutation_strength (float): Intensidad de la mutación.
        elite_size (int): Número de individuos elite a preservar.
        tournament_size (int): Tamaño del torneo para la selección.
        
    Returns:
        Tuple[List[Idea], List[float]]: Población final y lista de fitness promedio por generación.
    """
    # Inicializar población
    population = initialize_population(population_size, initializer_func)
    
    # Evaluar población inicial
    evaluate_population(population, evaluator_func)
    
    # Lista para almacenar el fitness promedio por generación
    avg_fitness_history = [sum(idea.fitness for idea in population) / len(population)]
    
    # Evolucionar por múltiples generaciones
    for generation in range(num_generations):
        population = evolve(
            population,
            evaluator_func,
            crossover_func,
            mutation_func,
            crossover_rate,
            mutation_rate,
            mutation_strength,
            elite_size,
            tournament_size
        )
        
        # Calcular y almacenar el fitness promedio de esta generación
        avg_fitness = sum(idea.fitness for idea in population) / len(population)
        avg_fitness_history.append(avg_fitness)
    
    return population, avg_fitness_history