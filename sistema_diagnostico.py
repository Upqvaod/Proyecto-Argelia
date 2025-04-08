import clips # Esta bibilioteca es para el motor de inferencia CLIPS
import spacy # Esta biblioteca es para el procesamiento de lenguaje natural
import re  # Esta biblioteca es para expresiones regulares 
import tkinter as tk # Estas biblioteca es para la interfaz gráfica de usuario
from tkinter import ttk, scrolledtext, messagebox
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
    "tos con flema": ["tos productiva", "tos con mucosidad", "expectoración"],
    "secreción nasal": ["goteo nasal", "rinorrea", "moco"],
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

# Definir reglas mejoradas utilizando el conocimiento del diccionario de sinónimos
env.build("""
(defrule detectar_covid
    (or (sintoma (nombre fiebre_alta))
        (sintoma (nombre temperatura_elevada))
        (sintoma (nombre hipertermia)))
    (or (sintoma (nombre tos_seca))
        (sintoma (nombre tos_persistente))
        (sintoma (nombre tos_improductiva)))
    (or (sintoma (nombre dificultad_para_respirar))
        (sintoma (nombre falta_de_aire))
        (sintoma (nombre disnea)))
    (or (sintoma (nombre perdida_de_olfato))
        (sintoma (nombre anosmia))
        (sintoma (nombre perdida_de_gusto))
        (sintoma (nombre ageusia)))
    =>
    (assert (diagnostico (enfermedad "COVID19") (certeza 95) (recomendacion "Aislamiento, prueba PCR y monitoreo médico inmediato"))))
""")

env.build("""
(defrule detectar_covid_parcial
    (or (sintoma (nombre fiebre_alta))
        (sintoma (nombre temperatura_elevada))
        (sintoma (nombre hipertermia)))
    (or (sintoma (nombre tos_seca))
        (sintoma (nombre tos_persistente))
        (sintoma (nombre tos_improductiva)))
    (or (sintoma (nombre dificultad_para_respirar))
        (sintoma (nombre falta_de_aire))
        (sintoma (nombre disnea)))
    =>
    (assert (diagnostico (enfermedad "COVID19") (certeza 85) (recomendacion "Aislamiento preventivo, prueba PCR y monitoreo de síntomas"))))
""")

env.build("""
(defrule detectar_gripe
    (or (sintoma (nombre fiebre))
        (sintoma (nombre temperatura))
        (sintoma (nombre calentura)))
    (or (sintoma (nombre dolor_muscular))
        (sintoma (nombre mialgia))
        (sintoma (nombre cuerpo_cortado)))
    (or (sintoma (nombre congestion_nasal))
        (sintoma (nombre nariz_tapada))
        (sintoma (nombre estornudos)))
    (or (sintoma (nombre dolor_de_cabeza))
        (sintoma (nombre cefalea))
        (sintoma (nombre escalofrios)))
    =>
    (assert (diagnostico (enfermedad "Gripe") (certeza 90) (recomendacion "Reposo, hidratación adecuada y analgésicos"))))
""")

env.build("""
(defrule detectar_gripe_parcial
    (or (sintoma (nombre fiebre))
        (sintoma (nombre temperatura))
        (sintoma (nombre calentura)))
    (or (sintoma (nombre dolor_muscular))
        (sintoma (nombre mialgia))
        (sintoma (nombre cuerpo_cortado)))
    (or (sintoma (nombre congestion_nasal))
        (sintoma (nombre nariz_tapada))
        (sintoma (nombre estornudos)))
    =>
    (assert (diagnostico (enfermedad "Gripe") (certeza 80) (recomendacion "Reposo, hidratación y analgésicos"))))
""")

env.build("""
(defrule detectar_neumonia
    (or (sintoma (nombre fiebre_alta))
        (sintoma (nombre temperatura_elevada))
        (sintoma (nombre hipertermia)))
    (or (sintoma (nombre tos_con_flema))
        (sintoma (nombre tos_productiva))
        (sintoma (nombre expectoracion)))
    (or (sintoma (nombre dolor_en_el_pecho)) 
        (sintoma (nombre dolor_toracico))
        (sintoma (nombre presion_en_el_pecho)))
    (or (sintoma (nombre dificultad_para_respirar))
        (sintoma (nombre falta_de_aire))
        (sintoma (nombre disnea)))
    =>
    (assert (diagnostico (enfermedad "Neumonia") (certeza 90) (recomendacion "Consulta médica urgente, posibles antibióticos y radiografía de tórax"))))
""")

env.build("""
(defrule detectar_bronquitis
    (or (sintoma (nombre tos_con_flema))
        (sintoma (nombre tos_productiva))
        (sintoma (nombre expectoracion)))
    (or (sintoma (nombre fatiga))
        (sintoma (nombre cansancio))
        (sintoma (nombre debilidad)))
    (or (sintoma (nombre fiebre))
        (sintoma (nombre temperatura))
        (sintoma (nombre calentura)))
    =>
    (assert (diagnostico (enfermedad "Bronquitis") (certeza 80) (recomendacion "Reposo, hidratación y posibles broncodilatadores"))))
""")

env.build("""
(defrule detectar_resfriado_comun
    (or (sintoma (nombre congestion_nasal))
        (sintoma (nombre nariz_tapada))
        (sintoma (nombre rinorrea)))
    (or (sintoma (nombre estornudos))
        (sintoma (nombre estornudar)))
    (or (sintoma (nombre dolor_de_garganta))
        (sintoma (nombre irritacion_de_garganta))
        (sintoma (nombre faringitis)))
    (or (sintoma (nombre secrecion_nasal))
        (sintoma (nombre goteo_nasal))
        (sintoma (nombre moco)))
    (not (sintoma (nombre fiebre_alta)))
    (not (sintoma (nombre dificultad_para_respirar)))
    =>
    (assert (diagnostico (enfermedad "Resfriado_Comun") (certeza 85) (recomendacion "Descanso, hidratación y medicamentos para aliviar síntomas"))))
""")

env.build("""
(defrule detectar_ataque_asma
    (or (sintoma (nombre dificultad_para_respirar))
        (sintoma (nombre falta_de_aire))
        (sintoma (nombre disnea))
        (sintoma (nombre ahogo)))
    (or (sintoma (nombre tos_seca))
        (sintoma (nombre tos_persistente))
        (sintoma (nombre tos_improductiva)))
    (historial (condicion "asma"))
    =>
    (assert (diagnostico (enfermedad "Ataque_Asma") (certeza 95) (recomendacion "Uso inmediato de inhalador de rescate y consulta médica urgente"))))
""")
    
env.build("""
(defrule detectar_ataque_asma_sin_historial
    (or (sintoma (nombre dificultad_para_respirar))
        (sintoma (nombre falta_de_aire))
        (sintoma (nombre disnea))
        (sintoma (nombre ahogo)))
    (or (sintoma (nombre tos_seca))
        (sintoma (nombre tos_persistente))
        (sintoma (nombre tos_improductiva)))
    (not (historial (condicion "asma")))
    =>
    (assert (diagnostico (enfermedad "Posible_Ataque_Asma") (certeza 70) (recomendacion "Buscar atención médica urgente para evaluación respiratoria"))))
""")

env.build("""
(defrule detectar_sinusitis
    (or (sintoma (nombre congestion_nasal))
        (sintoma (nombre nariz_tapada))
        (sintoma (nombre obstruccion_nasal)))
    (or (sintoma (nombre dolor_de_cabeza))
        (sintoma (nombre cefalea))
        (sintoma (nombre migrania)))
    (or (sintoma (nombre secrecion_nasal))
        (sintoma (nombre goteo_nasal))
        (sintoma (nombre rinorrea))
        (sintoma (nombre moco)))
    =>
    (assert (diagnostico (enfermedad "Sinusitis") (certeza 85) (recomendacion "Descongestionantes, analgésicos y consulta médica si persiste más de 7 días"))))
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
    (or (historial (condicion "asma"))
        (historial (condicion "enfermedad_respiratoria_cronica")))
    =>
    (assert (riesgo (factor "respiratorio") (nivel "alto"))))
""")

env.build("""
(defrule evaluar_riesgo_comorbilidades
    (or (historial (condicion "diabetes"))
        (historial (condicion "hipertension"))
        (historial (condicion "obesidad"))
        (historial (condicion "enfermedad_cardiaca")))
    =>
    (assert (riesgo (factor "comorbilidades") (nivel "alto"))))
""")

env.build("""
(defrule recomendaciones_prevencion_alto_riesgo
    (or (riesgo (factor "edad") (nivel "alto"))
        (riesgo (factor "respiratorio") (nivel "alto"))
        (riesgo (factor "comorbilidades") (nivel "alto")))
    =>
    (assert (diagnostico (enfermedad "Riesgo_alto_enfermedades_respiratorias") (certeza 100) (recomendacion "Vacunación completa, evitar aglomeraciones y uso de mascarilla en espacios cerrados"))))
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
     
     # Importancia relativa de cada síntoma (ponderación) para cada enfermedad
     ponderacion_sintomas = {
          "COVID19": {"fiebre_alta": 0.4, "tos_seca": 0.3, "dificultad_para_respirar": 0.3},
          "Gripe": {"fiebre": 0.4, "dolor_muscular": 0.3, "congestion_nasal": 0.3},
          "Neumonia": {"fiebre_alta": 0.3, "tos_con_flema": 0.3, "dolor_en_el_pecho": 0.2, "dificultad_para_respirar": 0.2},
          "Bronquitis": {"tos_con_flema": 0.5, "fatiga": 0.3, "fiebre": 0.2},
          "Resfriado_Comun": {"congestion_nasal": 0.4, "estornudos": 0.3, "dolor_de_garganta": 0.3},
          "Ataque_Asma": {"dificultad_para_respirar": 0.6, "tos_seca": 0.4},
          "Sinusitis": {"congestion_nasal": 0.4, "dolor_de_cabeza": 0.3, "secrecion_nasal": 0.3},
     }
     
     if enfermedad_objetivo not in requisitos_sintomas:
          return {
               "posible": False,
               "mensaje": "Enfermedad no reconocida en la base de conocimientos",
               "sintomas_faltantes": [],
               "certeza": 0,
               "justificacion": []
          }
     
     # Obtener síntomas actuales en los hechos
     sintomas_actuales = []
     for fact in env.facts():
          if fact.template.name == "sintoma":
               sintomas_actuales.append(fact["nombre"])
     
     # Comprobar síntomas requeridos y calcular certeza parcial
     sintomas_requeridos = requisitos_sintomas[enfermedad_objetivo]
     sintomas_faltantes = [s for s in sintomas_requeridos if s not in sintomas_actuales]
     sintomas_presentes = [s for s in sintomas_requeridos if s in sintomas_actuales]
     
     # Determinar el nivel de certeza basado en la ponderación de los síntomas presentes
     certeza = 0
     justificacion = []
     
     if enfermedad_objetivo in ponderacion_sintomas:
          pesos = ponderacion_sintomas[enfermedad_objetivo]
          for sintoma in sintomas_presentes:
               if sintoma in pesos:
                    peso = pesos[sintoma]
                    certeza += peso * 100
                    justificacion.append(f"Síntoma '{sintoma}' contribuye con {peso*100:.0f}% de certeza")
     
     # Si hay síntomas faltantes, explicar qué se necesitaría para confirmar
     explicacion_faltantes = []
     if sintomas_faltantes and enfermedad_objetivo in ponderacion_sintomas:
          for sintoma in sintomas_faltantes:
               if sintoma in ponderacion_sintomas[enfermedad_objetivo]:
                    peso = ponderacion_sintomas[enfermedad_objetivo][sintoma]
                    explicacion_faltantes.append(f"El síntoma '{sintoma}' aumentaría la certeza en {peso*100:.0f}%")
     
     # Determinar condiciones especiales que pueden influir en el diagnóstico
     condiciones_especiales = []
     historial_actual = []
     for fact in env.facts():
          if fact.template.name == "historial":
               historial_actual.append(fact["condicion"])
               
     if enfermedad_objetivo == "Ataque_Asma" and "asma" in historial_actual:
          certeza += 20
          condiciones_especiales.append("Historial de asma aumenta la probabilidad (+20%)")
     
     # Agregar explicación al diagnóstico
     return {
          "posible": certeza > 70,  # Se considera posible si la certeza supera el 70%
          "mensaje": "Diagnóstico posible" if certeza > 70 else "Certeza insuficiente para confirmar",
          "sintomas_faltantes": sintomas_faltantes,
          "explicaciones_faltantes": explicacion_faltantes,
          "certeza": min(certeza, 100),  # Asegurar que no supere el 100%
          "justificacion": justificacion + condiciones_especiales
     }

def forward_chaining_trace(sintomas, edad, historial):
     """Realiza encadenamiento hacia adelante con trazabilidad del razonamiento"""
     env.reset()
     
     # Rastrear hechos iniciales
     trace = {
          "hechos_iniciales": {
               "sintomas": sintomas,
               "edad": edad,
               "historial": historial
          },
          "pasos_inferencia": [],
          "diagnosticos": []
     }
     
     # Assert facts
     for sintoma in sintomas:
          env.assert_string(f'(sintoma (nombre {sintoma}))')
          trace["pasos_inferencia"].append(f"Afirmado síntoma: {sintoma}")
     
     env.assert_string(f'(edad (valor {edad}))')
     trace["pasos_inferencia"].append(f"Afirmada edad: {edad}")
     
     for condicion in historial:
          env.assert_string(f'(historial (condicion {condicion.replace(" ", "_")}))')
          trace["pasos_inferencia"].append(f"Afirmada condición: {condicion}")
     
     # Track rules fired
     prev_fact_count = len(list(env.facts()))
     
     env.run()
     
     # Track new facts generated
     curr_facts = list(env.facts())
     new_facts = curr_facts[prev_fact_count:]
     
     for fact in curr_facts:
          if fact.template.name == "diagnostico":
               trace["diagnosticos"].append({
                    "enfermedad": fact["enfermedad"],
                    "certeza": fact["certeza"],
                    "recomendacion": fact["recomendacion"]
               })
               trace["pasos_inferencia"].append(f"Generado diagnóstico: {fact['enfermedad']} con certeza {fact['certeza']}%")
          elif fact.template.name == "riesgo":
               trace["pasos_inferencia"].append(f"Identificado riesgo: {fact['factor']} nivel {fact['nivel']}")
     
     return trace

def explicar_diagnostico(enfermedad):
     """Proporciona una explicación detallada para un diagnóstico específico"""
     explicaciones = {
          "COVID19": "Enfermedad viral altamente contagiosa causada por el coronavirus SARS-CoV-2, que afecta principalmente al sistema respiratorio.",
          "Gripe": "Infección viral respiratoria causada por el virus influenza, caracterizada por fiebre, dolor muscular y congestión.",
          "Neumonia": "Infección que inflama los sacos aéreos en uno o ambos pulmones, causada por bacterias, virus u hongos.",
          "Bronquitis": "Inflamación de los bronquios, los conductos de aire que conectan la tráquea con los pulmones.",
          "Resfriado_Comun": "Infección viral leve del tracto respiratorio superior que afecta la nariz y garganta.",
          "Ataque_Asma": "Episodio agudo de obstrucción de las vías respiratorias debido a la inflamación y estrechamiento de las mismas.",
          "Sinusitis": "Inflamación de los senos paranasales, cavidades llenas de aire en el cráneo, generalmente debido a una infección.",
          "Riesgo_alto_enfermedades_respiratorias": "Factores identificados que aumentan la probabilidad de desarrollar o complicar enfermedades respiratorias."
     }
     
     if enfermedad in explicaciones:
          return explicaciones[enfermedad]
     return f"No hay explicación detallada disponible para {enfermedad}"

def diagnosticar_completo(texto, edad, historial, enfermedad_objetivo=None):
     """Integra encadenamiento hacia adelante y hacia atrás con explicaciones detalladas"""
     env.reset()
     
     # Procesar texto para extraer síntomas
     sintomas = extraer_sintomas(texto)
     print(f"\nSíntomas detectados: {sintomas}")
     
     # Realizar forward chaining con trazabilidad
     forward_trace = forward_chaining_trace(sintomas, edad, historial)
     
     resultados = {
          "sintomas_detectados": sintomas,
          "forward_chaining": forward_trace,
          "diagnosticos": []
     }
     
     # Si hay una enfermedad objetivo, usar encadenamiento hacia atrás
     if enfermedad_objetivo and enfermedad_objetivo != "ninguna":
          resultado_backward = backward_chaining(enfermedad_objetivo)
          resultados["backward_chaining"] = resultado_backward
          
          print(f"\nAnálisis específico para {enfermedad_objetivo}:")
          if resultado_backward["posible"]:
               print(f"✓ {resultado_backward['mensaje']} (Certeza: {resultado_backward['certeza']:.1f}%)")
               print(f"✓ {explicar_diagnostico(enfermedad_objetivo)}")
               for justif in resultado_backward["justificacion"]:
                    print(f"  - {justif}")
          else:
               print(f"✗ {resultado_backward['mensaje']}")
               if resultado_backward['sintomas_faltantes']:
                    print(f"Síntomas faltantes: {', '.join(resultado_backward['sintomas_faltantes'])}")
                    for expl in resultado_backward.get("explicaciones_faltantes", []):
                         print(f"  - {expl}")
     
     # Recopilar diagnósticos generados por forward chaining
     diagnosticos = []
     for diag in forward_trace["diagnosticos"]:
          diagnosticos.append({
               "enfermedad": diag["enfermedad"], 
               "certeza": diag["certeza"], 
               "recomendacion": diag["recomendacion"],
               "explicacion": explicar_diagnostico(diag["enfermedad"]),
               "razonamiento": [paso for paso in forward_trace["pasos_inferencia"] 
                              if diag["enfermedad"] in paso or "síntoma" in paso]
          })
     
     resultados["diagnosticos"] = diagnosticos
     return diagnosticos

#################################################
# PARTE 3: INTERFAZ GRÁFICA CON TKINTER
#################################################

class DiagnosticoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Experto de Diagnóstico Médico")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")
        
        # Crear frame con scrollbar
        self.container = ttk.Frame(root)
        self.container.pack(fill=tk.BOTH, expand=True)
        
        # Crear canvas con scrollbar vertical
        self.canvas = tk.Canvas(self.container, bg="#f0f0f0")
        self.scrollbar = ttk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configurar el scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        # Añadir el frame al canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Colocar canvas y scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Configurar scrolling con rueda del ratón
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Crear contenedor principal
        main_frame = ttk.Frame(self.scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(main_frame, text="Sistema Experto de Diagnóstico Médico", 
                  font=("Helvetica", 16, "bold")).pack(pady=0)
        
        # Frame para entrada de datos
        input_frame = ttk.LabelFrame(main_frame, text="Datos del Paciente", padding="10")
        input_frame.pack(fill=tk.X, pady=2)
        
        # Descripción de síntomas
        ttk.Label(input_frame, text="Describa sus síntomas:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.sintomas_text = scrolledtext.ScrolledText(input_frame, height=5, width=70, wrap=tk.WORD)
        self.sintomas_text.grid(row=1, column=0, columnspan=2, pady=5, padx=5, sticky=tk.W+tk.E)
        
        # Edad
        ttk.Label(input_frame, text="Edad:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.edad_var = tk.StringVar(value="30")
        edad_spinbox = ttk.Spinbox(input_frame, from_=0, to=120, textvariable=self.edad_var, width=5)
        edad_spinbox.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Condiciones preexistentes
        cond_frame = ttk.LabelFrame(main_frame, text="Condiciones Preexistentes", padding="10")
        cond_frame.pack(fill=tk.X, pady=5)
        
        self.cond_vars = {
            "asma": tk.BooleanVar(),
            "diabetes": tk.BooleanVar(),
            "hipertension": tk.BooleanVar(),
            "obesidad": tk.BooleanVar(),
            "enfermedad_cardiaca": tk.BooleanVar(),
            "enfermedad_respiratoria_cronica": tk.BooleanVar()
        }
        
        # Crear checkboxes para condiciones
        for i, (cond, var) in enumerate(self.cond_vars.items()):
            row = i // 3
            col = i % 3
            ttk.Checkbutton(cond_frame, text=cond.replace("_", " ").title(), 
                           variable=var).grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
        
        # Análisis específico
        enferm_frame = ttk.LabelFrame(main_frame, text="Análisis Específico", padding="10")
        enferm_frame.pack(fill=tk.X, pady=1)
        
        self.enfermedad_var = tk.StringVar(value="ninguna")
        enfermedades = {
            "ninguna": "Análisis general",
            "COVID19": "COVID-19",
            "Gripe": "Gripe",
            "Neumonia": "Neumonía",
            "Bronquitis": "Bronquitis",
            "Resfriado_Comun": "Resfriado Común",
            "Ataque_Asma": "Ataque de Asma",
            "Sinusitis": "Sinusitis",
            "Riesgo_alto_enfermedades_respiratorias": "Riesgo alto enfermedades respiratorias"
        }
        
        for i, (val, text) in enumerate(enfermedades.items()):
            ttk.Radiobutton(enferm_frame, text=text, value=val, 
                           variable=self.enfermedad_var).grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=5)
        
        # Botones de acción
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Diagnosticar", command=self.realizar_diagnostico).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Limpiar", command=self.limpiar_campos).pack(side=tk.LEFT, padx=10)
        
        # Área de resultados
        result_frame = ttk.LabelFrame(main_frame, text="Resultados del Diagnóstico", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.resultado_text = scrolledtext.ScrolledText(result_frame, height=20, wrap=tk.WORD)
        self.resultado_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def _on_mousewheel(self, event):
        """Maneja el evento de desplazamiento con la rueda del ratón"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def limpiar_campos(self):
        """Limpia todos los campos del formulario"""
        self.sintomas_text.delete("1.0", tk.END)
        self.edad_var.set("30")
        for var in self.cond_vars.values():
            var.set(False)
        self.enfermedad_var.set("ninguna")
        self.resultado_text.delete("1.0", tk.END)
    
    def realizar_diagnostico(self):
        """Realiza el diagnóstico con los datos ingresados"""
        try:
            # Obtener datos del formulario
            descripcion = self.sintomas_text.get("1.0", tk.END).strip()
            if not descripcion:
                messagebox.showwarning("Datos incompletos", "Por favor describa sus síntomas")
                return
            
            try:
                edad = int(self.edad_var.get())
            except ValueError:
                messagebox.showwarning("Valor inválido", "La edad debe ser un número entero")
                return
            
            historial = [cond for cond, var in self.cond_vars.items() if var.get()]
            
            enfermedad_objetivo = self.enfermedad_var.get()
            if enfermedad_objetivo == "ninguna":
                enfermedad_objetivo = None
            
            # Realizar diagnóstico
            self.resultado_text.delete("1.0", tk.END)
            self.resultado_text.insert(tk.INSERT, "Procesando diagnóstico...\n\n")
            self.root.update()
            
            # Detección de síntomas
            sintomas = extraer_sintomas(descripcion)
            self.resultado_text.insert(tk.INSERT, f"Síntomas detectados: {', '.join(sintomas)}\n\n")
            
            # Análisis
            diagnosticos = diagnosticar_completo(descripcion, edad, historial, enfermedad_objetivo)
            
            # Mostrar resultados
            if diagnosticos:
                self.resultado_text.insert(tk.INSERT, "=== Diagnóstico completado ===\n\n")
                for diag in diagnosticos:
                    self.resultado_text.insert(tk.INSERT, f"• {diag['enfermedad']} (Certeza: {diag['certeza']}%)\n")
                    self.resultado_text.insert(tk.INSERT, f"  Explicación: {diag['explicacion']}\n")
                    self.resultado_text.insert(tk.INSERT, f"  Recomendación: {diag['recomendacion']}\n\n")
            else:
                self.resultado_text.insert(tk.INSERT, "No se pudo generar un diagnóstico con los síntomas proporcionados.\n")
                self.resultado_text.insert(tk.INSERT, "Intente describir sus síntomas con mayor detalle.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en el diagnóstico: {str(e)}")
            self.resultado_text.insert(tk.INSERT, f"Error: {str(e)}")

#################################################
# PARTE 4: EJECUCIÓN PRINCIPAL 
#################################################

if __name__ == "__main__":
    try:
        # Iniciar la aplicación GUI
        root = tk.Tk()
        app = DiagnosticoApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error al iniciar la aplicación: {str(e)}")
