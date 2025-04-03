import clips
import spacy

# Load existing environment setup
from sistema_experto import env, extraer_sintomas, normalizar_sintomas, sinonimos_sintomas

# Add backward chaining capability
def backward_chaining(enfermedad_objetivo):
    """
    Implementa razonamiento hacia atrás para verificar si una enfermedad específica
    puede ser diagnosticada con base en los síntomas actuales.
    """
    # Obtener reglas que concluyen la enfermedad objetivo
    requisitos_sintomas = {
        "COVID19": ["fiebre_alta", "tos_seca", "dificultad_para_respirar"],
        "Gripe": ["fiebre", "dolor_muscular", "congestion_nasal"],
        "Riesgo_alto_enfermedades_respiratorias": []  # Este depende de factores de riesgo, no síntomas
    }
    
    if enfermedad_objetivo not in requisitos_sintomas:
        return {
            "posible": False,
            "mensaje": "Enfermedad no reconocida en la base de conocimientos",
            "sintomas_faltantes": []
        }
    
    # Obtener síntomas actuales en los hechos
    sintomas_actuales = []
    for fact in env.facts():
        if fact.template.name == "sintoma":
            sintomas_actuales.append(fact["nombre"])
    
    # Comprobar si todos los síntomas necesarios están presentes
    sintomas_requeridos = requisitos_sintomas[enfermedad_objetivo]
    sintomas_faltantes = [s for s in sintomas_requeridos if s not in sintomas_actuales]
    
    return {
        "posible": len(sintomas_faltantes) == 0,
        "mensaje": "Diagnóstico posible" if len(sintomas_faltantes) == 0 else "Faltan síntomas para confirmar",
        "sintomas_faltantes": sintomas_faltantes
    }

def explicar_diagnostico(enfermedad):
    """
    Explica cómo se llegó a un diagnóstico específico, mostrando las reglas involucradas.
    """
    explicaciones = {
        "COVID19": "El diagnóstico de COVID-19 se basa en la presencia de fiebre alta, tos seca, y dificultad para respirar.",
        "Gripe": "El diagnóstico de gripe se basa en la presencia de fiebre, dolor muscular, y congestión nasal.",
        "Riesgo_alto_enfermedades_respiratorias": "Este aviso se genera cuando hay un factor de riesgo alto, como la edad avanzada."
    }
    
    return explicaciones.get(enfermedad, "No hay explicación disponible para esta condición.")

# Ejemplo de uso combinado de encadenamiento hacia adelante y hacia atrás
def diagnosticar_completo(texto, edad, historial, enfermedad_objetivo=None):
    """Integra encadenamiento hacia adelante y hacia atrás"""
    env.reset()
    
    # Procesar texto para extraer síntomas (encadenamiento hacia adelante)
    sintomas = extraer_sintomas(texto)
    print(f"\nSíntomas detectados: {sintomas}")
    
    for sintoma in sintomas:
        env.assert_string(f'(sintoma (nombre {sintoma}))')
    
    env.assert_string(f'(edad (valor {edad}))')
    
    for condicion in historial:
        env.assert_string(f'(historial (condicion {condicion.replace(" ", "_")}))')
    
    # Si hay una enfermedad objetivo, usar encadenamiento hacia atrás primero
    if enfermedad_objetivo:
        resultado_backward = backward_chaining(enfermedad_objetivo)
        if resultado_backward["posible"]:
            print(f"\nAnálisis específico para {enfermedad_objetivo}:")
            print(f"✓ {resultado_backward['mensaje']}")
            print(f"✓ {explicar_diagnostico(enfermedad_objetivo)}")
        else:
            print(f"\nAnálisis específico para {enfermedad_objetivo}:")
            print(f"✗ {resultado_backward['mensaje']}")
            if resultado_backward['sintomas_faltantes']:
                print(f"Síntomas faltantes: {', '.join(resultado_backward['sintomas_faltantes'])}")
    
    # Ejecutar motor de inferencia (encadenamiento hacia adelante)
    env.run()
    
    # Recopilar diagnósticos generados
    diagnosticos = []
    for fact in env.facts():
        if fact.template.name == "diagnostico":
            diagnosticos.append({
                "enfermedad": fact["enfermedad"], 
                "certeza": fact["certeza"], 
                "recomendacion": fact["recomendacion"],
                "explicacion": explicar_diagnostico(fact["enfermedad"])
            })
    
    return diagnosticos

# Demo de uso
if __name__ == "__main__":
    print("=== Sistema de diagnóstico médico con razonamiento bidireccional ===")
    print("\n1. Encadenamiento hacia adelante (automático):")
    texto = "Tengo fiebre alta, tos persistente y me cuesta respirar"
    edad = 65
    historial = ["asma"]
    
    resultados = diagnosticar_completo(texto, edad, historial)
    for r in resultados:
        print(f"\n- {r['enfermedad']} (Certeza: {r['certeza']}%)")
        print(f"  Explicación: {r['explicacion']}")
        print(f"  Recomendación: {r['recomendacion']}")
    
    print("\n2. Encadenamiento hacia atrás (comprobando hipótesis específica):")
    resultados = diagnosticar_completo(texto, edad, historial, "COVID19")