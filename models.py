"""
Módulo que contiene la clase Idea para representar el cromosoma del algoritmo genético.
Cada Idea consiste en un problema y una solución.
"""

class Idea:
    """
    Clase que representa una idea de negocio como un cromosoma genético.
    
    Atributos:
        problem (str): Descripción del problema que la idea intenta resolver.
        solution (str): Descripción de la solución propuesta para el problema.
        fitness (float): Puntuación de aptitud de la idea (más alta = mejor).
    """
    
    def __init__(self, problem="", solution="", fitness=0.0):
        """
        Inicializa una nueva instancia de Idea.
        
        Args:
            problem (str): Descripción del problema. Por defecto, cadena vacía.
            solution (str): Descripción de la solución. Por defecto, cadena vacía.
            fitness (float): Puntuación inicial de aptitud. Por defecto, 0.0.
        """
        self.problem = problem
        self.solution = solution
        self.fitness = fitness
    
    def __str__(self):
        """
        Devuelve una representación en cadena de texto de la idea.
        
        Returns:
            str: Representación legible de la idea.
        """
        return f"Problema: {self.problem}\nSolución: {self.solution}\nAptitud: {self.fitness:.2f}"
    
    def __repr__(self):
        """
        Devuelve una representación formal de la idea.
        
        Returns:
            str: Representación formal de la idea.
        """
        return f"Idea(problem='{self.problem}', solution='{self.solution}', fitness={self.fitness})"
    
    def copy(self):
        """
        Crea una copia de esta idea.
        
        Returns:
            Idea: Una nueva instancia de Idea con los mismos atributos.
        """
        return Idea(problem=self.problem, solution=self.solution, fitness=self.fitness)