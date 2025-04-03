import clips
import spacy

# Cargar modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# Inicializar entorno CLIPS
env = clips.Environment()

# Diccionario de sinónimos para mejorar el reconocimiento de síntomas
sinonimos_sintomas = {
    "fiebre alta": ["temperatura elevada", "calentura"],
    "tos seca": ["tos persistente"],
    "dolor de cabeza": ["cefalea", "migraña"],
    "congestión nasal": ["nariz tapada"],
    "dificultad para respirar": ["falta de aire", "disnea"],
    "dolor muscular": ["mialgia", "cuerpo cortado"],
    "estornudos": ["estornudar"],
    "mareo": ["vértigo"],
    "dolor abdominal": ["malestar estomacal"],
}

# Definir la base de conocimientos en CLIPS
env.build("""
   (deftemplate sintoma (slot nombre))
   (deftemplate edad (slot valor))
   (deftemplate historial (slot condicion))
   (deftemplate diagnostico (slot enfermedad) (slot certeza) (slot recomendacion))
   (deftemplate riesgo (slot factor) (slot nivel))
""")

env.build("""
   (defrule detectar_covid
      (sintoma (nombre fiebre_alta))
      (sintoma (nombre tos_seca))
      (sintoma (nombre dificultad_para_respirar))
      =>
      (assert (diagnostico (enfermedad "COVID19") (certeza 90) (recomendacion "Aislamiento, prueba PCR y monitoreo médico."))))
""")

env.build("""
   (defrule detectar_gripe
      (sintoma (nombre fiebre))
      (sintoma (nombre dolor_muscular))
      (sintoma (nombre congestion_nasal))
      =>
      (assert (diagnostico (enfermedad "Gripe") (certeza 80) (recomendacion "Reposo, hidratación y analgésicos."))))
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
      (assert (diagnostico (enfermedad "Riesgo_alto_enfermedades_respiratorias") (certeza 100) (recomendacion "Vacunación y evitar aglomeraciones."))))
""")

def normalizar_sintomas(sintoma):
    """Convierte un síntoma en su versión estandarizada usando sinónimos"""
    for clave, sinonimos in sinonimos_sintomas.items():
        if sintoma in sinonimos:
            return clave.replace(" ", "_")
    return sintoma.replace(" ", "_")

def extraer_sintomas(texto):
    """Extrae síntomas relevantes y normaliza con sinónimos"""
    doc = nlp(texto.lower())
    sintomas_detectados = [normalizar_sintomas(token.text) for token in doc if token.pos_ in ["NOUN", "ADJ"]]
    return list(set(sintomas_detectados))

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

# Prueba del sistema
texto_usuario = "Tengo fiebre alta, tos persistente y me cuesta respirar."
edad_usuario = 65
historial_usuario = ["asma"]

diagnosticos = diagnosticar(texto_usuario, edad_usuario, historial_usuario)
print("Diagnóstico:")
for enfermedad, certeza, recomendacion in diagnosticos:
    print(f"- {enfermedad} ({certeza}%) → {recomendacion}")
