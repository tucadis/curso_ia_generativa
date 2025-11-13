# Módulo 6: Ejemplos Prácticos

## Contenido del Módulo

1. [Proyecto 1: Generación de Texto con APIs](#proyecto-1-generación-de-texto-con-apis)
2. [Proyecto 2: Chatbot Personalizado](#proyecto-2-chatbot-personalizado)
3. [Proyecto 3: Generación de Imágenes](#proyecto-3-generación-de-imágenes)
4. [Proyecto 4: RAG - Retrieval Augmented Generation](#proyecto-4-rag---retrieval-augmented-generation)
5. [Proyecto 5: Fine-tuning de Modelos](#proyecto-5-fine-tuning-de-modelos)
6. [Proyecto 6: Aplicación Web con IA](#proyecto-6-aplicación-web-con-ia)

---

## Proyecto 1: Generación de Texto con APIs

### Objetivo
Crear un generador de contenido usando la API de OpenAI.

### Requisitos
```bash
pip install openai python-dotenv
```

### Paso 1: Configuración

Crea un archivo `.env`:
```
OPENAI_API_KEY=tu-api-key-aqui
```

### Paso 2: Código Base

```python
# text_generator.py
import openai
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generar_texto(prompt, modelo="gpt-3.5-turbo", temperatura=0.7, max_tokens=500):
    """
    Genera texto usando la API de OpenAI.

    Args:
        prompt (str): El prompt o instrucción
        modelo (str): Modelo a usar
        temperatura (float): Creatividad (0-2)
        max_tokens (int): Longitud máxima de respuesta

    Returns:
        str: Texto generado
    """
    try:
        response = openai.ChatCompletion.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Eres un asistente útil y creativo."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperatura,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# Ejemplo de uso
if __name__ == "__main__":
    # Generación de blog post
    prompt = """
    Escribe un artículo de blog de 300 palabras sobre
    'El impacto de la IA Generativa en el marketing digital'.
    Incluye introducción, 3 puntos principales y conclusión.
    """

    resultado = generar_texto(prompt, temperatura=0.7)
    print(resultado)
```

### Paso 3: Casos de Uso Específicos

#### A) Generador de Emails

```python
def generar_email(tema, tono="profesional", destinatario="cliente"):
    """Genera un email personalizado."""

    prompt = f"""
    Escribe un email {tono} para un {destinatario} sobre: {tema}

    Estructura:
    - Saludo apropiado
    - Cuerpo del mensaje (3-4 párrafos)
    - Llamado a la acción
    - Despedida
    """

    return generar_texto(prompt, temperatura=0.5)

# Uso
email = generar_email(
    tema="Lanzamiento de nuevo producto de IA",
    tono="entusiasta",
    destinatario="cliente potencial"
)
print(email)
```

#### B) Generador de Código

```python
def generar_codigo(descripcion, lenguaje="Python"):
    """Genera código según descripción."""

    prompt = f"""
    Genera código en {lenguaje} para: {descripcion}

    Incluye:
    - Código completo y funcional
    - Comentarios explicativos
    - Manejo de errores
    - Ejemplo de uso
    """

    return generar_texto(prompt, temperatura=0.3, max_tokens=1000)

# Uso
codigo = generar_codigo(
    "función que valida y formatea direcciones de email"
)
print(codigo)
```

#### C) Traductor con Contexto

```python
def traducir_con_contexto(texto, idioma_origen, idioma_destino, contexto=""):
    """Traduce texto considerando contexto."""

    prompt = f"""
    Traduce el siguiente texto de {idioma_origen} a {idioma_destino}.

    Contexto: {contexto}

    Texto a traducir:
    {texto}

    Proporciona:
    1. Traducción directa
    2. Traducción adaptada culturalmente
    3. Notas sobre diferencias importantes
    """

    return generar_texto(prompt, temperatura=0.3)

# Uso
traduccion = traducir_con_contexto(
    texto="This AI model is killing it!",
    idioma_origen="inglés",
    idioma_destino="español",
    contexto="Expresión coloquial en tecnología"
)
print(traduccion)
```

### Paso 4: Mejoras y Optimizaciones

#### Manejo de Errores Robusto

```python
import time
from openai.error import RateLimitError, APIError

def generar_texto_robusto(prompt, max_reintentos=3):
    """Genera texto con reintentos automáticos."""

    for intento in range(max_reintentos):
        try:
            return generar_texto(prompt)

        except RateLimitError:
            if intento < max_reintentos - 1:
                tiempo_espera = 2 ** intento  # Exponential backoff
                print(f"Rate limit alcanzado. Esperando {tiempo_espera}s...")
                time.sleep(tiempo_espera)
            else:
                raise

        except APIError as e:
            print(f"Error de API: {e}")
            if intento < max_reintentos - 1:
                time.sleep(1)
            else:
                raise
```

---

## Proyecto 2: Chatbot Personalizado

### Objetivo
Crear un chatbot con memoria que mantiene contexto de la conversación.

### Código Completo

```python
# chatbot.py
import openai
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Chatbot:
    """Chatbot con memoria y personalidad."""

    def __init__(self, nombre="Asistente", personalidad="amigable y servicial"):
        self.nombre = nombre
        self.personalidad = personalidad
        self.historial = []
        self.sistema_prompt = f"""
        Eres {nombre}, un asistente {personalidad}.

        Características:
        - Respondes de manera clara y concisa
        - Mantienes el contexto de la conversación
        - Admites cuando no sabes algo
        - Eres respetuoso y profesional
        """

        # Agregar mensaje del sistema
        self.historial.append({
            "role": "system",
            "content": self.sistema_prompt
        })

    def chat(self, mensaje_usuario):
        """Procesa mensaje del usuario y genera respuesta."""

        # Agregar mensaje del usuario al historial
        self.historial.append({
            "role": "user",
            "content": mensaje_usuario
        })

        try:
            # Generar respuesta
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.historial,
                temperature=0.7,
                max_tokens=500
            )

            # Extraer respuesta
            respuesta = response.choices[0].message.content

            # Agregar respuesta al historial
            self.historial.append({
                "role": "assistant",
                "content": respuesta
            })

            return respuesta

        except Exception as e:
            return f"Error: {str(e)}"

    def limpiar_historial(self):
        """Reinicia la conversación manteniendo el sistema prompt."""
        self.historial = [self.historial[0]]  # Solo mantener sistema

    def guardar_conversacion(self, archivo="conversacion.txt"):
        """Guarda el historial de conversación."""
        with open(archivo, "w", encoding="utf-8") as f:
            f.write(f"Conversación con {self.nombre}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")

            for msg in self.historial[1:]:  # Saltar sistema
                rol = "Usuario" if msg["role"] == "user" else self.nombre
                f.write(f"{rol}: {msg['content']}\n\n")

# Interfaz de línea de comandos
def main():
    """Ejecuta el chatbot en modo interactivo."""

    print("=== Chatbot IA ===")
    print("Escribe 'salir' para terminar")
    print("Escribe 'limpiar' para reiniciar conversación")
    print("Escribe 'guardar' para guardar la conversación\n")

    # Crear chatbot
    bot = Chatbot(
        nombre="Claude Junior",
        personalidad="experto en IA y programación"
    )

    while True:
        # Obtener input del usuario
        mensaje = input("Tú: ").strip()

        if not mensaje:
            continue

        # Comandos especiales
        if mensaje.lower() == "salir":
            print("¡Hasta luego!")
            break

        elif mensaje.lower() == "limpiar":
            bot.limpiar_historial()
            print("Conversación reiniciada.\n")
            continue

        elif mensaje.lower() == "guardar":
            bot.guardar_conversacion()
            print("Conversación guardada.\n")
            continue

        # Generar respuesta
        respuesta = bot.chat(mensaje)
        print(f"\n{bot.nombre}: {respuesta}\n")

if __name__ == "__main__":
    main()
```

### Extensión: Chatbot con Funciones

```python
import json

class ChatbotAvanzado(Chatbot):
    """Chatbot que puede ejecutar funciones."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.funciones_disponibles = {
            "obtener_clima": self.obtener_clima,
            "calcular": self.calcular,
            "buscar_web": self.buscar_web
        }

    def obtener_clima(self, ciudad):
        """Simula obtener el clima (placeholder)."""
        return f"El clima en {ciudad} es soleado, 22°C"

    def calcular(self, expresion):
        """Evalúa expresión matemática."""
        try:
            resultado = eval(expresion)
            return f"Resultado: {resultado}"
        except:
            return "Error en el cálculo"

    def buscar_web(self, consulta):
        """Simula búsqueda web (placeholder)."""
        return f"Resultados para '{consulta}': [simulado]"

    def chat_con_funciones(self, mensaje):
        """Chat que puede llamar funciones."""

        # Aquí puedes implementar lógica para detectar
        # cuándo llamar funciones basándote en el mensaje

        # Por simplicidad, llamamos chat normal
        return self.chat(mensaje)
```

---

## Proyecto 3: Generación de Imágenes

### Objetivo
Generar imágenes usando diferentes APIs.

### Opción A: DALL-E (OpenAI)

```python
# image_generator.py
import openai
import os
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generar_imagen_dalle(prompt, n=1, tamaño="1024x1024"):
    """
    Genera imagen usando DALL-E.

    Args:
        prompt (str): Descripción de la imagen
        n (int): Número de imágenes
        tamaño (str): Tamaño (256x256, 512x512, 1024x1024)

    Returns:
        list: URLs de imágenes generadas
    """
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=tamaño
        )

        urls = [img['url'] for img in response['data']]
        return urls

    except Exception as e:
        print(f"Error: {e}")
        return []

def descargar_y_mostrar(url, nombre_archivo="imagen.png"):
    """Descarga imagen desde URL y la muestra."""

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Guardar
    img.save(nombre_archivo)
    print(f"Imagen guardada como {nombre_archivo}")

    # Mostrar
    img.show()

    return img

# Ejemplo de uso
if __name__ == "__main__":
    prompt = """
    Un gato astronauta flotando en el espacio,
    con la Tierra al fondo, estilo digital art,
    colores vibrantes, alta calidad
    """

    print("Generando imagen...")
    urls = generar_imagen_dalle(prompt)

    if urls:
        print(f"Imagen generada: {urls[0]}")
        descargar_y_mostrar(urls[0])
```

### Opción B: Stable Diffusion Local

```python
# stable_diffusion_local.py
from diffusers import StableDiffusionPipeline
import torch

def inicializar_modelo(modelo_id="runwayml/stable-diffusion-v1-5"):
    """Carga el modelo de Stable Diffusion."""

    # Verificar si hay GPU disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cargar pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        modelo_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    return pipe

def generar_imagen_sd(pipe, prompt, negative_prompt="", steps=50, guidance=7.5):
    """
    Genera imagen con Stable Diffusion.

    Args:
        pipe: Pipeline de Stable Diffusion
        prompt (str): Descripción de lo que quieres
        negative_prompt (str): Lo que NO quieres
        steps (int): Pasos de denoising
        guidance (float): Adherencia al prompt

    Returns:
        PIL.Image: Imagen generada
    """
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]

    return image

# Ejemplo de uso
if __name__ == "__main__":
    print("Cargando modelo (puede tardar)...")
    pipe = inicializar_modelo()

    prompt = "a beautiful sunset over mountains, highly detailed, 4k"
    negative_prompt = "blurry, low quality, distorted"

    print("Generando imagen...")
    imagen = generar_imagen_sd(pipe, prompt, negative_prompt)

    # Guardar
    imagen.save("salida.png")
    imagen.show()
    print("Imagen guardada como salida.png")
```

### Generador de Variaciones

```python
def generar_variaciones_dalle(imagen_path, n=3):
    """Genera variaciones de una imagen existente."""

    with open(imagen_path, "rb") as f:
        response = openai.Image.create_variation(
            image=f,
            n=n,
            size="1024x1024"
        )

    return [img['url'] for img in response['data']]
```

---

## Proyecto 4: RAG - Retrieval Augmented Generation

### Objetivo
Crear un sistema que responde preguntas basándose en tus propios documentos.

### Instalación

```bash
pip install langchain openai chromadb pypdf tiktoken
```

### Código Completo

```python
# rag_system.py
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

class SistemaRAG:
    """Sistema RAG para Q&A sobre documentos propios."""

    def __init__(self, directorio_documentos, persist_directory="./chroma_db"):
        self.directorio_documentos = directorio_documentos
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None

    def cargar_documentos(self):
        """Carga documentos desde directorio."""

        print("Cargando documentos...")

        # Cargar PDFs
        loader_pdf = DirectoryLoader(
            self.directorio_documentos,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        # Cargar TXT
        loader_txt = DirectoryLoader(
            self.directorio_documentos,
            glob="**/*.txt",
            loader_cls=TextLoader
        )

        docs_pdf = loader_pdf.load()
        docs_txt = loader_txt.load()

        documentos = docs_pdf + docs_txt
        print(f"Cargados {len(documentos)} documentos")

        return documentos

    def dividir_documentos(self, documentos):
        """Divide documentos en chunks."""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_documents(documentos)
        print(f"Creados {len(chunks)} chunks")

        return chunks

    def crear_vectorstore(self, chunks):
        """Crea base de datos vectorial."""

        print("Creando embeddings...")

        embeddings = OpenAIEmbeddings()

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )

        self.vectorstore.persist()
        print("Vectorstore creado y guardado")

    def cargar_vectorstore(self):
        """Carga vectorstore existente."""

        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        print("Vectorstore cargado")

    def crear_qa_chain(self):
        """Crea cadena de Q&A."""

        # Template de prompt
        template = """
        Usa el siguiente contexto para responder la pregunta al final.
        Si no sabes la respuesta, di que no lo sabes, NO inventes información.
        Usa máximo 3 oraciones y mantén la respuesta concisa.

        Contexto: {context}

        Pregunta: {question}

        Respuesta útil:
        """

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )

        # Crear chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        print("Chain Q&A creado")

    def inicializar(self, recrear=False):
        """Inicializa el sistema completo."""

        if recrear or not os.path.exists(self.persist_directory):
            # Crear desde cero
            documentos = self.cargar_documentos()
            chunks = self.dividir_documentos(documentos)
            self.crear_vectorstore(chunks)
        else:
            # Cargar existente
            self.cargar_vectorstore()

        self.crear_qa_chain()
        print("Sistema RAG listo!")

    def preguntar(self, pregunta):
        """Hace una pregunta al sistema."""

        if not self.qa_chain:
            raise Exception("Sistema no inicializado. Llama a inicializar() primero.")

        resultado = self.qa_chain({"query": pregunta})

        return {
            "respuesta": resultado["result"],
            "fuentes": resultado["source_documents"]
        }

# Uso del sistema
if __name__ == "__main__":
    # Crear sistema
    rag = SistemaRAG(directorio_documentos="./mis_documentos")

    # Inicializar (recrear=True para procesar documentos nuevos)
    rag.inicializar(recrear=False)

    # Modo interactivo
    print("\n=== Sistema RAG de Preguntas y Respuestas ===")
    print("Escribe 'salir' para terminar\n")

    while True:
        pregunta = input("Pregunta: ").strip()

        if pregunta.lower() == "salir":
            break

        if not pregunta:
            continue

        # Obtener respuesta
        resultado = rag.preguntar(pregunta)

        print(f"\nRespuesta: {resultado['respuesta']}\n")

        # Mostrar fuentes (opcional)
        print("Fuentes:")
        for i, doc in enumerate(resultado['fuentes'], 1):
            print(f"{i}. {doc.metadata.get('source', 'Desconocido')}")
        print()
```

### Uso Avanzado: RAG con Historial

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class SistemaRAGConversacional(SistemaRAG):
    """RAG que mantiene historial de conversación."""

    def crear_qa_chain(self):
        """Crea cadena conversacional."""

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )

    def preguntar(self, pregunta):
        """Pregunta considerando historial."""

        resultado = self.qa_chain({"question": pregunta})

        return {
            "respuesta": resultado["answer"],
            "fuentes": resultado["source_documents"]
        }
```

---

## Proyecto 5: Fine-tuning de Modelos

### Objetivo
Ajustar un modelo preentrenado para una tarea específica.

### Fine-tuning de GPT-3.5 (OpenAI)

```python
# fine_tuning_openai.py
import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def preparar_datos_entrenamiento(ejemplos, archivo_salida="training_data.jsonl"):
    """
    Prepara datos en formato JSONL para fine-tuning.

    Args:
        ejemplos (list): Lista de diccionarios con 'prompt' y 'completion'
    """
    with open(archivo_salida, 'w') as f:
        for ejemplo in ejemplos:
            json_line = json.dumps({
                "messages": [
                    {"role": "system", "content": "Eres un asistente experto en..."},
                    {"role": "user", "content": ejemplo["prompt"]},
                    {"role": "assistant", "content": ejemplo["completion"]}
                ]
            })
            f.write(json_line + '\n')

    print(f"Datos guardados en {archivo_salida}")

# Ejemplo: Crear dataset para clasificación de sentimientos
ejemplos_entrenamiento = [
    {
        "prompt": "Este producto es increíble, lo recomiendo totalmente",
        "completion": "Positivo"
    },
    {
        "prompt": "Muy decepcionado con la compra, no funciona",
        "completion": "Negativo"
    },
    {
        "prompt": "Es un producto normal, nada especial",
        "completion": "Neutral"
    },
    # ... más ejemplos (mínimo 10, recomendado 50+)
]

# Preparar datos
preparar_datos_entrenamiento(ejemplos_entrenamiento)

def subir_archivo_entrenamiento(archivo_path):
    """Sube archivo de entrenamiento a OpenAI."""

    with open(archivo_path, 'rb') as f:
        response = openai.File.create(
            file=f,
            purpose='fine-tune'
        )

    print(f"Archivo subido: {response['id']}")
    return response['id']

def iniciar_fine_tuning(file_id, modelo_base="gpt-3.5-turbo"):
    """Inicia proceso de fine-tuning."""

    response = openai.FineTuningJob.create(
        training_file=file_id,
        model=modelo_base
    )

    print(f"Fine-tuning iniciado: {response['id']}")
    return response['id']

def verificar_estado(job_id):
    """Verifica estado del fine-tuning."""

    response = openai.FineTuningJob.retrieve(job_id)
    print(f"Estado: {response['status']}")
    return response

def usar_modelo_finetuned(modelo_id, prompt):
    """Usa el modelo fine-tuned."""

    response = openai.ChatCompletion.create(
        model=modelo_id,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Proceso completo
if __name__ == "__main__":
    # 1. Preparar datos
    preparar_datos_entrenamiento(ejemplos_entrenamiento)

    # 2. Subir archivo
    file_id = subir_archivo_entrenamiento("training_data.jsonl")

    # 3. Iniciar fine-tuning
    job_id = iniciar_fine_tuning(file_id)

    # 4. Esperar (puede tardar minutos u horas)
    # Verificar estado periódicamente
    # estado = verificar_estado(job_id)

    # 5. Una vez completado, usar modelo
    # resultado = usar_modelo_finetuned("ft:gpt-3.5-turbo:...", "Este producto es excelente")
    # print(resultado)
```

### Fine-tuning con Hugging Face

```python
# fine_tuning_hf.py
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

def fine_tune_clasificador():
    """Fine-tuning de modelo para clasificación."""

    # 1. Cargar modelo base
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # Positivo/Negativo
    )

    # 2. Cargar dataset
    dataset = load_dataset("imdb")

    # 3. Tokenizar
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Configurar entrenamiento
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 5. Crear Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"].shuffle().select(range(1000)),
        eval_dataset=tokenized_datasets["test"].shuffle().select(range(200)),
    )

    # 6. Entrenar
    trainer.train()

    # 7. Guardar modelo
    model.save_pretrained("./mi_modelo_finetuned")
    tokenizer.save_pretrained("./mi_modelo_finetuned")

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = fine_tune_clasificador()
```

---

## Proyecto 6: Aplicación Web con IA

### Objetivo
Crear una aplicación web interactiva con Streamlit.

### Instalación

```bash
pip install streamlit openai python-dotenv
```

### Código Completo

```python
# app.py
import streamlit as st
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuración de página
st.set_page_config(
    page_title="Asistente IA Multiherramienta",
    page_icon=">",
    layout="wide"
)

# Título
st.title("> Asistente IA Multiherramienta")
st.markdown("---")

# Sidebar con opciones
st.sidebar.title("Herramientas")
opcion = st.sidebar.radio(
    "Selecciona una herramienta:",
    ["Chat", "Generador de Texto", "Resumen", "Traductor", "Análisis de Sentimiento"]
)

# Función auxiliar para llamar a OpenAI
def llamar_gpt(prompt, system_message="Eres un asistente útil.", temp=0.7):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=temp
    )
    return response.choices[0].message.content

# === HERRAMIENTA 1: CHAT ===
if opcion == "Chat":
    st.header("=¬ Chat con IA")

    # Inicializar historial en session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Escribe tu mensaje..."):
        # Agregar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                respuesta = llamar_gpt(prompt)
                st.markdown(respuesta)

        # Agregar respuesta al historial
        st.session_state.messages.append({"role": "assistant", "content": respuesta})

# === HERRAMIENTA 2: GENERADOR DE TEXTO ===
elif opcion == "Generador de Texto":
    st.header("
 Generador de Texto")

    col1, col2 = st.columns([2, 1])

    with col1:
        tipo_texto = st.selectbox(
            "Tipo de contenido:",
            ["Blog Post", "Email", "Publicación Redes Sociales", "Descripción de Producto"]
        )

        tema = st.text_input("Tema o descripción:")
        tono = st.select_slider("Tono:", ["Formal", "Neutral", "Casual", "Entusiasta"])

    with col2:
        longitud = st.slider("Longitud (palabras):", 50, 500, 200)
        temperatura = st.slider("Creatividad:", 0.0, 1.5, 0.7)

    if st.button("Generar", type="primary"):
        if tema:
            with st.spinner("Generando contenido..."):
                prompt = f"""
                Escribe un {tipo_texto.lower()} sobre: {tema}

                Requisitos:
                - Tono: {tono}
                - Aproximadamente {longitud} palabras
                - Bien estructurado y atractivo
                """

                resultado = llamar_gpt(prompt, temp=temperatura)

                st.markdown("### Resultado:")
                st.write(resultado)

                # Botón para copiar
                st.code(resultado, language=None)
        else:
            st.warning("Por favor ingresa un tema")

# === HERRAMIENTA 3: RESUMEN ===
elif opcion == "Resumen":
    st.header("=Ý Resumidor de Texto")

    texto_largo = st.text_area("Pega aquí el texto a resumir:", height=300)

    col1, col2 = st.columns(2)
    with col1:
        longitud_resumen = st.select_slider(
            "Longitud del resumen:",
            ["Muy breve", "Breve", "Medio", "Detallado"]
        )
    with col2:
        formato = st.radio("Formato:", ["Párrafo", "Puntos clave"])

    if st.button("Resumir", type="primary"):
        if texto_largo:
            with st.spinner("Analizando y resumiendo..."):
                prompt = f"""
                Resume el siguiente texto de forma {longitud_resumen.lower()}.
                Formato: {'Lista de puntos clave' if formato == 'Puntos clave' else 'Párrafo cohesivo'}

                Texto:
                {texto_largo}
                """

                resumen = llamar_gpt(prompt, temp=0.3)

                st.markdown("### Resumen:")
                st.info(resumen)

                # Estadísticas
                col1, col2, col3 = st.columns(3)
                col1.metric("Palabras originales", len(texto_largo.split()))
                col2.metric("Palabras resumen", len(resumen.split()))
                col3.metric("Reducción", f"{int((1 - len(resumen.split())/len(texto_largo.split()))*100)}%")
        else:
            st.warning("Por favor ingresa un texto")

# === HERRAMIENTA 4: TRADUCTOR ===
elif opcion == "Traductor":
    st.header("< Traductor IA")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Texto Original")
        idioma_origen = st.selectbox("Idioma origen:", ["Español", "Inglés", "Francés", "Alemán", "Portugués"])
        texto_origen = st.text_area("Texto a traducir:", height=200)

    with col2:
        st.subheader("Traducción")
        idioma_destino = st.selectbox("Idioma destino:", ["Inglés", "Español", "Francés", "Alemán", "Portugués"])

    if st.button("Traducir", type="primary"):
        if texto_origen:
            with st.spinner("Traduciendo..."):
                prompt = f"""
                Traduce el siguiente texto de {idioma_origen} a {idioma_destino}.
                Mantén el tono y estilo del original.

                Texto:
                {texto_origen}
                """

                traduccion = llamar_gpt(prompt, temp=0.3)

                with col2:
                    st.text_area("Resultado:", traduccion, height=200)
        else:
            st.warning("Por favor ingresa un texto")

# === HERRAMIENTA 5: ANÁLISIS DE SENTIMIENTO ===
elif opcion == "Análisis de Sentimiento":
    st.header("=
 Análisis de Sentimiento")

    texto_analizar = st.text_area("Texto a analizar:", height=150)

    if st.button("Analizar", type="primary"):
        if texto_analizar:
            with st.spinner("Analizando sentimiento..."):
                prompt = f"""
                Analiza el sentimiento del siguiente texto.

                Proporciona:
                1. Sentimiento general (Positivo/Negativo/Neutral)
                2. Puntuación de 0-10
                3. Emociones detectadas
                4. Tono del mensaje

                Formato tu respuesta claramente.

                Texto:
                {texto_analizar}
                """

                analisis = llamar_gpt(prompt, temp=0.3)

                st.markdown("### Análisis:")
                st.success(analisis)
        else:
            st.warning("Por favor ingresa un texto")

# Footer
st.markdown("---")
st.markdown("Desarrollado con d usando Streamlit y OpenAI")
```

### Ejecutar la Aplicación

```bash
streamlit run app.py
```

---

##  Resumen del Módulo

En este módulo has creado:

 **Generador de texto** con OpenAI API
 **Chatbot** con memoria y personalidad
 **Generador de imágenes** con DALL-E y Stable Diffusion
 **Sistema RAG** para Q&A sobre documentos propios
 **Fine-tuning** de modelos para tareas específicas
 **Aplicación web** completa con múltiples herramientas

---

## Recursos del Módulo

Todos los códigos están disponibles en la carpeta `/ejemplos` del repositorio.

---

## Ejercicios Propuestos

1. Personaliza el chatbot con tu propia personalidad
2. Crea un RAG sobre documentación técnica de tu interés
3. Agrega una nueva herramienta a la aplicación web
4. Experimenta con diferentes temperaturas y parámetros
5. Combina múltiples herramientas en un workflow

---

## ¡ Próximo Módulo

Ahora explorarás **aplicaciones reales** de IA Generativa en diferentes industrias.

[Módulo 7: Aplicaciones en la Industria](../modulo7/README.md)

[ Volver al Módulo 5](../modulo5/README.md) | [ Volver al inicio](../../README.md)
