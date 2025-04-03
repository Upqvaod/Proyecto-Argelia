import clips
import spacy
import tkinter as tk
from tkinter import ttk, scrolledtext

###########################################
# PARTE 1: SISTEMA EXPERTO BASE (CLIPS)
###########################################

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
# PARTE 3: INTERFAZ GRÁFICA (TKINTER)
#################################################

class DiagnosticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Experto de Diagnóstico Médico")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        # Configurar estilo
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 11))
        style.configure("TButton", font=("Arial", 11))
        
        # Marco principal
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(main_frame, text="Sistema Experto de Diagnóstico Médico", 
                 font=("Arial", 16, "bold")).pack(pady=10)
        
        # Marco de entrada
        input_frame = ttk.Frame(main_frame, padding="10")
        input_frame.pack(fill=tk.X, pady=5)  # <-- This was missing!
        
        # Descripción de síntomas
        ttk.Label(input_frame, text="Describa sus síntomas:").pack(anchor=tk.W)
        
        self.symptom_text = scrolledtext.ScrolledText(input_frame, height=5, wrap=tk.WORD)
        self.symptom_text.pack(fill=tk.X, pady=5)
        self.symptom_text.insert(tk.END, "Ejemplo: Tengo fiebre alta, tos persistente y dificultad para respirar")
        
        # Edad
        age_frame = ttk.Frame(input_frame)
        age_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(age_frame, text="Edad:").pack(side=tk.LEFT)
        
        self.age_var = tk.StringVar(value="50")
        age_entry = ttk.Entry(age_frame, textvariable=self.age_var, width=5)
        age_entry.pack(side=tk.LEFT, padx=5)
        
        # Historial médico
        ttk.Label(input_frame, text="Historial médico (seleccione condiciones preexistentes):").pack(anchor=tk.W, pady=(10,0))
        
        history_frame = ttk.Frame(input_frame)
        history_frame.pack(fill=tk.X)
        
        self.asthma_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(history_frame, text="Asma", variable=self.asthma_var).pack(side=tk.LEFT, padx=5)
        
        self.diabetes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(history_frame, text="Diabetes", variable=self.diabetes_var).pack(side=tk.LEFT, padx=5)
        
        self.hypertension_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(history_frame, text="Hipertensión", variable=self.hypertension_var).pack(side=tk.LEFT, padx=5)
        
        # Enfermedad a comprobar específicamente (para encadenamiento hacia atrás)
        ttk.Label(input_frame, text="Enfermedad a verificar (opcional, para encadenamiento hacia atrás):").pack(anchor=tk.W, pady=(10,0))
        
        self.disease_var = tk.StringVar()
        disease_combo = ttk.Combobox(input_frame, textvariable=self.disease_var)
        disease_combo['values'] = ('', 'COVID19', 'Gripe', 'Riesgo_alto_enfermedades_respiratorias')
        disease_combo.pack(fill=tk.X, pady=5)
        
        # Botón de diagnóstico
        ttk.Button(input_frame, text="Realizar diagnóstico", command=self.perform_diagnosis).pack(pady=10)
        
        # Marco de resultados
        result_frame = ttk.Frame(main_frame, padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(result_frame, text="Resultados:").pack(anchor=tk.W)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
    def perform_diagnosis(self):
        """Realizar diagnóstico usando el sistema experto"""
        
        # Limpiar resultados anteriores
        self.result_text.delete(1.0, tk.END)
        
        # Obtener datos de entrada
        symptoms_text = self.symptom_text.get(1.0, tk.END).strip()
        
        try:
            age = int(self.age_var.get())
        except ValueError:
            self.result_text.insert(tk.END, "Error: Por favor ingrese una edad válida\n")
            return
        
        # Construir historial médico
        history = []
        if self.asthma_var.get():
            history.append("asma")
        if self.diabetes_var.get():
            history.append("diabetes")
        if self.hypertension_var.get():
            history.append("hipertension")
        
        # Obtener enfermedad objetivo (si está seleccionada)
        target_disease = self.disease_var.get() if self.disease_var.get() else None
        
        # Llamar a nuestro sistema experto
        try:
            diagnosticos = diagnosticar_completo(symptoms_text, age, history, target_disease)
            
            # Mostrar resultados
            if diagnosticos:
                self.result_text.insert(tk.END, "=== Diagnóstico completado ===\n\n")
                for diag in diagnosticos:
                    self.result_text.insert(tk.END, f"• {diag['enfermedad']} (Certeza: {diag['certeza']}%)\n")
                    self.result_text.insert(tk.END, f"  Explicación: {diag['explicacion']}\n")
                    self.result_text.insert(tk.END, f"  Recomendación: {diag['recomendacion']}\n\n")
            else:
                self.result_text.insert(tk.END, "No se pudo generar un diagnóstico con los síntomas proporcionados.\n" +
                                      "Intente describir sus síntomas con mayor detalle.\n")
        except Exception as e:
            self.result_text.insert(tk.END, f"Error en el diagnóstico: {str(e)}\n")

#################################################
# PARTE 4: EJECUCIÓN PRINCIPAL
#################################################

if __name__ == "__main__":
    # Ejecutar la interfaz gráfica
    root = tk.Tk()
    app = DiagnosticApp(root)
    root.mainloop()