import clips
import spacy
import re
from collections import defaultdict

###########################################
# PARTE 1: SISTEMA EXPERTO BASE (CLIPS)
###########################################

# Cargar modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# Inicializar entorno CLIPS
env = clips.Environment()

# Diccionario extendido de sinónimos para mejorar el reconocimiento de síntomas
sinonimos_sintomas = {
    "fiebre alta": ["temperatura elevada", "calentura", "pirexia", "hipertermia"],
    "fiebre": ["temperatura", "calentura"],
    "tos seca": ["tos persistente", "tos irritativa", "tos improductiva", "tos sin flema"],
    "tos": ["toser", "toses"],
    "dolor de cabeza": ["cefalea", "migraña", "jaqueca", "dolor craneal"],
    "congestión nasal": ["nariz tapada", "obstrucción nasal", "rinorrea"],
    "dificultad para respirar": ["falta de aire", "disnea", "ahogo", "respiración dificultosa"],
    "dolor muscular": ["mialgia", "cuerpo cortado", "dolores musculares", "dolor en músculos"],
    "estornudos": ["estornudar", "estornudo frecuente"],
    "mareo": ["vértigo", "aturdimiento", "desequilibrio"],
    "dolor abdominal": ["malestar estomacal", "dolor de estómago", "molestia abdominal"],
    "dolor de garganta": ["irritación de garganta", "faringitis", "molestia en la garganta"],
    "fatiga": ["cansancio", "debilidad", "agotamiento"],
    "pérdida de olfato": ["anosmia", "sin olfato", "no huele"],
    "pérdida de gusto": ["ageusia", "sin sabor", "no saborea"],
    "diarrea": ["heces sueltas", "deposiciones líquidas", "evacuaciones frecuentes"],
    "náusea": ["náuseas", "ganas de vomitar", "malestar estomacal", "asco"],
    "vómitos": ["vomitar", "devolver", "expulsar alimentos"],
    "dolor en el pecho": ["dolor torácico", "presión en el pecho", "molestia pectoral"],
    "erupción cutánea": ["sarpullido", "rash", "exantema", "erupciones en la piel"],
    "dolor articular": ["artralgia", "dolor en las articulaciones", "rigidez articular"],
    "glándulas inflamadas": ["ganglios inflamados", "adenopatía", "inflamación ganglionar"],
    "escalofríos": ["tiritona", "temblores por frío", "estremecimiento"],
    "sudoración": ["sudores", "transpiración excesiva", "hiperhidrosis"],
    "tos con flema": ["tos productiva", "tos con mucosidad", "expectoración"],
    "secreción nasal": ["goteo nasal", "rinorrea", "moco"],
    "insomnio": ["dificultad para dormir", "desvelo", "falta de sueño"],
    "pérdida de apetito": ["inapetencia", "anorexia", "falta de hambre"],
}

# Frases negativas que indican ausencia de síntomas
frases_negativas = ["no tengo", "sin", "ausencia de", "no presenta", "no hay", "negativo", 
                   "no padezco", "no sufro", "no siento", "no me duele"]

# Definir templates de forma individual
env.build("(deftemplate sintoma (slot nombre))")
env.build("(deftemplate edad (slot valor))")
env.build("(deftemplate historial (slot condicion))")
env.build("(deftemplate diagnostico (slot enfermedad) (slot certeza) (slot recomendacion))")
env.build("(deftemplate riesgo (slot factor) (slot nivel))")

# Definir reglas de forma individual
env.build("""
(defrule detectar_covid
    (sintoma (nombre fiebre_alta))
    (sintoma (nombre tos_seca))
    (sintoma (nombre dificultad_para_respirar))
    =>
    (assert (diagnostico (enfermedad "COVID19") (certeza 90) (recomendacion "Aislamiento, prueba PCR y monitoreo medico"))))
""")

env.build("""
(defrule detectar_gripe
    (sintoma (nombre fiebre))
    (sintoma (nombre dolor_muscular))
    (sintoma (nombre congestion_nasal))
    =>
    (assert (diagnostico (enfermedad "Gripe") (certeza 80) (recomendacion "Reposo, hidratacion y analgesicos"))))
""")

# Nuevas reglas para enfermedades respiratorias
env.build("""
(defrule detectar_neumonia
    (sintoma (nombre fiebre_alta))
    (sintoma (nombre tos_con_flema))
    (sintoma (nombre dolor_en_el_pecho))
    (sintoma (nombre dificultad_para_respirar))
    =>
    (assert (diagnostico (enfermedad "Neumonia") (certeza 85) (recomendacion "Consulta medica urgente, posibles antibioticos y radiografia de torax"))))
""")

env.build("""
(defrule detectar_bronquitis
    (sintoma (nombre tos_con_flema))
    (sintoma (nombre fatiga))
    (sintoma (nombre fiebre))
    =>
    (assert (diagnostico (enfermedad "Bronquitis") (certeza 75) (recomendacion "Reposo, hidratacion y posibles broncodilatadores"))))
""")

env.build("""
(defrule detectar_resfriado_comun
    (sintoma (nombre congestion_nasal))
    (sintoma (nombre estornudos))
    (sintoma (nombre dolor_de_garganta))
    (not (sintoma (nombre fiebre_alta)))
    =>
    (assert (diagnostico (enfermedad "Resfriado_Comun") (certeza 70) (recomendacion "Descanso, hidratacion y remedios sintomaticos"))))
""")

env.build("""
(defrule detectar_ataque_asma
    (sintoma (nombre dificultad_para_respirar))
    (sintoma (nombre tos_seca))
    (historial (condicion "asma"))
    =>
    (assert (diagnostico (enfermedad "Ataque_Asma") (certeza 85) (recomendacion "Uso de inhalador de rescate y consulta medica"))))
""")

env.build("""
(defrule detectar_sinusitis
    (sintoma (nombre congestion_nasal))
    (sintoma (nombre dolor_de_cabeza))
    (sintoma (nombre secrecion_nasal))
    =>
    (assert (diagnostico (enfermedad "Sinusitis") (certeza 75) (recomendacion "Descongestionantes, analgésicos y posible consulta médica"))))
""")

env.build("""
(defrule evaluar_riesgo_edad
    (edad (valor ?e))
    (test (>= ?e 60))
    =>
    (assert (riesgo (factor "edad") (nivel "alto"))))
""")

env.build("""
(defrule evaluar_riesgo_historial
    (historial (condicion "asma"))
    =>
    (assert (riesgo (factor "asma") (nivel "medio"))))
""")

env.build("""
(defrule recomendaciones_prevencion
    (riesgo (factor "edad") (nivel "alto"))
    =>
    (assert (diagnostico (enfermedad "Riesgo_alto_enfermedades_respiratorias") (certeza 100) (recomendacion "Vacunacion y evitar aglomeraciones"))))
""")

def es_negacion(texto, sintoma):
     """Determina si un síntoma está negado en el texto"""
     for negacion in frases_negativas:
          if re.search(rf'{negacion}\b\s+\w+\s*({sintoma})', texto, re.IGNORECASE):
                return True
     return False

def normalizar_sintomas(token, texto_completo):
     """Convierte un síntoma en su versión estandarizada usando sinónimos"""
     # Comprobar si es una negación
     for clave, sinonimos in sinonimos_sintomas.items():
          if token.text in sinonimos or token.text in clave:
                if es_negacion(texto_completo, token.text):
                     return None  # Ignoramos los síntomas negados
                return clave.replace(" ", "_")
          
     # Para detección de frases completas
     for clave, sinonimos in sinonimos_sintomas.items():
          for frase in [clave] + sinonimos:
                if frase in texto_completo and not es_negacion(texto_completo, frase):
                     return clave.replace(" ", "_")
                     
     return token.text.replace(" ", "_")

def extraer_sintomas(texto):
     """Extrae síntomas relevantes con análisis lingüístico mejorado"""
     doc = nlp(texto.lower())
     
     # Detectar frases completas primero
     sintomas_detectados = set()
     for clave, sinonimos in sinonimos_sintomas.items():
          for frase in [clave] + sinonimos:
                if frase in texto.lower() and not es_negacion(texto.lower(), frase):
                     sintomas_detectados.add(clave.replace(" ", "_"))
     
     # Usar análisis gramatical para detectar síntomas basados en sustantivos y adjetivos
     for token in doc:
          if token.pos_ in ["NOUN", "ADJ"]:
                sintoma = normalizar_sintomas(token, texto.lower())
                if sintoma:
                     sintomas_detectados.add(sintoma)
     
     # Detectar relaciones entre palabras usando dependencias sintácticas
     for token in doc:
          if token.dep_ in ["amod", "dobj", "nsubj"] and token.head.text in sintomas_detectados:
                combinacion = f"{token.head.text}_{token.text}".replace(" ", "_")
                sintoma = normalizar_sintomas(token, texto.lower())
                if sintoma:
                     sintomas_detectados.add(sintoma)
     
     return list(sintomas_detectados)

def diagnosticar(texto, edad, historial):
     """Procesa el texto ingresado, extrae síntomas y ejecuta el motor de inferencia en CLIPS."""
     env.reset()
     
     sintomas = extraer_sintomas(texto)
     for sintoma in sintomas:
          env.assert_string(f'(sintoma (nombre {sintoma}))')
     
     env.assert_string(f'(edad (valor {edad}))')
     
     for condicion in historial:
          env.assert_string(f'(historial (condicion {condicion.replace(" ", "_")}))')
     
     env.run()
     
     diagnosticos = []
     for fact in env.facts():
          if fact.template.name == "diagnostico":
                diagnosticos.append((fact["enfermedad"], fact["certeza"], fact["recomendacion"]))
     
     return diagnosticos

#################################################
# PARTE 2: RAZONAMIENTO (FORWARD & BACKWARD CHAINING)
#################################################

def backward_chaining(enfermedad_objetivo):
     """
     Implementa razonamiento hacia atrás para verificar si una enfermedad específica
     puede ser diagnosticada con base en los síntomas actuales.
     """
     # Obtener reglas que concluyen la enfermedad objetivo
     requisitos_sintomas = {
          "COVID19": ["fiebre_alta", "tos_seca", "dificultad_para_respirar"],
          "Gripe": ["fiebre", "dolor_muscular", "congestion_nasal"],
          "Neumonia": ["fiebre_alta", "tos_con_flema", "dolor_en_el_pecho", "dificultad_para_respirar"],
          "Bronquitis": ["tos_con_flema", "fatiga", "fiebre"],
          "Resfriado_Comun": ["congestion_nasal", "estornudos", "dolor_de_garganta"],
          "Ataque_Asma": ["dificultad_para_respirar", "tos_seca"],
          "Sinusitis": ["congestion_nasal", "dolor_de_cabeza", "secrecion_nasal"],
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
        "Neumonia": "El diagnóstico de neumonía se basa en la presencia de fiebre alta, tos con flema, dolor en el pecho y dificultad para respirar.",
        "Bronquitis": "El diagnóstico de bronquitis se basa en la presencia de tos con flema, fatiga y fiebre.",
        "Resfriado_Comun": "El diagnóstico de resfriado común se basa en la presencia de congestión nasal, estornudos y dolor de garganta.",
        "Ataque_Asma": "El diagnóstico de ataque de asma se basa en la presencia de dificultad para respirar y tos seca, especialmente en pacientes con historial de asma.",
        "Sinusitis": "El diagnóstico de sinusitis se basa en la presencia de congestión nasal, dolor de cabeza y secreción nasal.",
        "Riesgo_alto_enfermedades_respiratorias": "Este aviso se genera cuando hay un factor de riesgo alto, como la edad avanzada o condiciones preexistentes."
    }
    
    return explicaciones.get(enfermedad, "No hay explicación disponible para esta condición.")

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

#################################################
# PARTE 4: EJECUCIÓN PRINCIPAL 
#################################################

if __name__ == "__main__":
    print("Sistema Experto de Diagnóstico Médico")
    print("=====================================")
    
    try:
        # Recopilar información del usuario a través de la consola
        descripcion = input("Describa sus síntomas: ")
        
        try:
            edad = int(input("Introduzca su edad: "))
        except ValueError:
            print("Edad no válida. Se usará 30 por defecto.")
            edad = 30
        
        print("\nSeleccione condiciones preexistentes (introduzca números separados por comas):")
        print("1. Asma")
        print("2. Diabetes")
        print("3. Hipertensión")
        
        historial = []
        seleccion = input("Selección: ")
        if "1" in seleccion:
            historial.append("asma")
        if "2" in seleccion:
            historial.append("diabetes")
        if "3" in seleccion:
            historial.append("hipertension")
        
        print("\nEnfermedades disponibles para análisis específico:")
        print("1. COVID19")
        print("2. Gripe")
        print("3. Riesgo alto enfermedades respiratorias")
        print("0. Ninguna (análisis general)")
        
        enfermedad_objetivo = None
        seleccion_enfermedad = input("Seleccione una enfermedad a verificar (0-3): ")
        if seleccion_enfermedad == "1":
            enfermedad_objetivo = "COVID19"
        elif seleccion_enfermedad == "2":
            enfermedad_objetivo = "Gripe"
        elif seleccion_enfermedad == "3":
            enfermedad_objetivo = "Riesgo_alto_enfermedades_respiratorias"
        
        # Realizar diagnóstico
        diagnosticos = diagnosticar_completo(descripcion, edad, historial, enfermedad_objetivo)
        
        # Mostrar resultados
        print("\n=== Diagnóstico completado ===")
        if diagnosticos:
            for diag in diagnosticos:
                print(f"\n• {diag['enfermedad']} (Certeza: {diag['certeza']}%)")
                print(f"  Explicación: {diag['explicacion']}")
                print(f"  Recomendación: {diag['recomendacion']}")
        else:
            print("\nNo se pudo generar un diagnóstico con los síntomas proporcionados.")
            print("Intente describir sus síntomas con mayor detalle.")
            
    except Exception as e:
        print(f"Error en el diagnóstico: {str(e)}")
