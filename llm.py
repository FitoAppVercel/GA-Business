"""
Módulo para integración con modelos de lenguaje (LLMs) a través de Ollama.
Este módulo proporciona funciones para generar ideas de negocio utilizando modelos de lenguaje.
"""

import json
import re
import requests
import logging
from typing import Dict, Any, Optional, Tuple

from models import Idea

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def call_ollama(prompt: str, model: str = "mistral") -> str:
    """
    Realiza una llamada al API de Ollama para generar texto.
    
    Args:
        prompt (str): El texto de entrada para el modelo.
        model (str): El nombre del modelo a utilizar. Por defecto, "mistral".
        
    Returns:
        str: La respuesta generada por el modelo.
        
    Raises:
        Exception: Si ocurre un error en la comunicación con Ollama.
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Lanza una excepción si el status code es 4XX/5XX
        
        # Parseamos la respuesta JSON
        response_data = response.json()
        
        if "response" in response_data:
            return response_data["response"]
        else:
            logger.error(f"Respuesta inesperada de Ollama: {response_data}")
            return ""
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al comunicarse con Ollama: {e}")
        raise Exception(f"Error al comunicarse con Ollama: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar la respuesta JSON: {e}")
        raise Exception(f"Error al decodificar la respuesta JSON: {e}")

def parse_idea_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parsea el texto generado por el LLM para extraer el problema y la solución.
    
    Args:
        text (str): Texto generado por el LLM.
        
    Returns:
        Tuple[Optional[str], Optional[str]]: Una tupla (problema, solución) extraída del texto.
    """
    # Intentamos primero con regex para encontrar el patrón esperado
    problem_match = re.search(r'Problema\s*:\s*(.*?)(?=\nSolución|$)', text, re.DOTALL | re.IGNORECASE)
    solution_match = re.search(r'Solución\s*:\s*(.*?)(?=\n\w+:|$)', text, re.DOTALL | re.IGNORECASE)
    
    problem = problem_match.group(1).strip() if problem_match else None
    solution = solution_match.group(1).strip() if solution_match else None
    
    # Si no podemos extraer ambos, intentamos dividir por líneas y buscar manualmente
    if problem is None or solution is None:
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'problema' in line.lower() and ':' in line and i+1 < len(lines):
                problem = line.split(':', 1)[1].strip() if problem is None else problem
            elif 'solución' in line.lower() and ':' in line and i+1 < len(lines):
                solution = line.split(':', 1)[1].strip() if solution is None else solution
    
    return problem, solution

def generate_idea_with_llm() -> Idea:
    """
    Genera una idea de negocio utilizando un modelo de lenguaje.
    
    Returns:
        Idea: Una instancia de la clase Idea con el problema y solución generados.
    """
    prompt = "Genera una idea de negocio en formato:\nProblema: ...\nSolución: ..."
    
    try:
        # Llamar al modelo
        response_text = call_ollama(prompt)
        
        # Parsear la respuesta
        problem, solution = parse_idea_from_text(response_text)
        
        # Verificar si pudimos extraer el problema y la solución
        if problem and solution:
            return Idea(problem=problem, solution=solution)
        else:
            # Si no se pudo parsear correctamente, mostramos advertencia
            missing = []
            if not problem:
                missing.append("problema")
            if not solution:
                missing.append("solución")
            
            logger.warning(f"No se pudo extraer {' y '.join(missing)} de la respuesta del modelo")
            logger.warning(f"Respuesta original: {response_text}")
            
            # Crear una idea predeterminada como fallback
            return Idea(
                problem="Problema no detectado en la respuesta del modelo",
                solution="Solución no detectada en la respuesta del modelo"
            )
            
    except Exception as e:
        logger.error(f"Error al generar idea con el modelo: {e}")
        # Crear una idea predeterminada como fallback
        return Idea(
            problem="Error al comunicarse con el modelo de lenguaje",
            solution="Por favor, verifica que Ollama esté en ejecución"
        )