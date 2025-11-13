# Módulo 5: Lenguajes y Librerías para IA Generativa

## =Ë Contenido del Módulo

1. [Python para IA](#python-para-ia)
2. [Frameworks de Deep Learning](#frameworks-de-deep-learning)
3. [Hugging Face Ecosystem](#hugging-face-ecosystem)
4. [LangChain y Frameworks para LLMs](#langchain-y-frameworks-para-llms)
5. [APIs de Modelos Generativos](#apis-de-modelos-generativos)
6. [Herramientas y Utilidades](#herramientas-y-utilidades)

---

## Python para IA

Python es el **lenguaje dominante** para IA y Machine Learning.

### ¿Por qué Python?

 **Sintaxis simple**: Fácil de aprender y leer
 **Ecosistema rico**: Miles de librerías especializadas
 **Comunidad masiva**: Soporte y recursos abundantes
 **Integración**: Funciona con herramientas de bajo nivel (C/C++)
 **Notebooks**: Jupyter para experimentación interactiva

### Instalación y Setup

#### 1. Instalar Python

```bash
# Verificar instalación
python --version  # o python3 --version

# Debería ser Python 3.8+
```

**Descargar**: [python.org](https://www.python.org/downloads/)

#### 2. Entornos Virtuales

```bash
# Crear entorno virtual
python -m venv mi_proyecto_ia

# Activar (Linux/Mac)
source mi_proyecto_ia/bin/activate

# Activar (Windows)
mi_proyecto_ia\Scripts\activate

# Desactivar
deactivate
```

#### 3. Package Manager: pip

```bash
# Instalar paquete
pip install nombre-paquete

# Instalar versión específica
pip install nombre-paquete==1.2.3

# Ver paquetes instalados
pip list

# Guardar dependencias
pip freeze > requirements.txt

# Instalar desde requirements
pip install -r requirements.txt
```

### Librerías Fundamentales de Python

#### NumPy - Computación Numérica

```python
import numpy as np

# Crear array
array = np.array([1, 2, 3, 4, 5])

# Operaciones vectorizadas
array * 2  # [2, 4, 6, 8, 10]

# Matrices
matrix = np.array([[1, 2], [3, 4]])
```

**Instalación**: `pip install numpy`

**Para qué se usa**:
- Operaciones matemáticas rápidas
- Arrays multidimensionales
- Álgebra lineal

---

#### Pandas - Manipulación de Datos

```python
import pandas as pd

# Crear DataFrame
df = pd.DataFrame({
    'nombre': ['Ana', 'Luis', 'María'],
    'edad': [25, 30, 28]
})

# Leer CSV
df = pd.read_csv('datos.csv')

# Operaciones
df.describe()  # Estadísticas
df.groupby('columna').mean()  # Agregaciones
```

**Instalación**: `pip install pandas`

**Para qué se usa**:
- Análisis de datos
- Limpieza de datos
- Procesamiento de datasets

---

#### Matplotlib y Seaborn - Visualización

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico simple
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Visualización con seaborn
sns.heatmap(data, annot=True)
```

**Instalación**:
```bash
pip install matplotlib seaborn
```

**Para qué se usa**:
- Visualizar datos
- Gráficos de entrenamiento
- Análisis exploratorio

---

## Frameworks de Deep Learning

### 1. PyTorch

**Desarrollado por**: Meta (Facebook)

#### Características

 **Pythonic**: Sintaxis natural de Python
 **Dynamic computation graphs**: Flexible y debugging fácil
 **Research-friendly**: Preferido en investigación
 **Fuerte en NLP**: Base de Hugging Face

#### Instalación

```bash
# CPU
pip install torch

# GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Ejemplo Básico

```python
import torch
import torch.nn as nn

# Crear tensor
x = torch.tensor([1.0, 2.0, 3.0])

# Operaciones
y = x * 2
print(y)  # tensor([2., 4., 6.])

# Red neuronal simple
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleNN()
```

#### Componentes Clave

**Tensores**: Arrays multidimensionales con GPU support
```python
# CPU
tensor_cpu = torch.tensor([1, 2, 3])

# GPU
if torch.cuda.is_available():
    tensor_gpu = tensor_cpu.cuda()
```

**Autograd**: Diferenciación automática
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # dy/dx = 2x = 4.0
```

**nn.Module**: Construcción de modelos
```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

#### Ecosistema PyTorch

- **torchvision**: Visión por computadora
- **torchaudio**: Procesamiento de audio
- **torchtext**: NLP (menos usado ahora)
- **PyTorch Lightning**: Simplifica entrenamiento

---

### 2. TensorFlow / Keras

**Desarrollado por**: Google

#### Características

 **Producción-ready**: Excelente para deployment
 **TensorFlow Serving**: Infraestructura para producción
 **Keras**: API de alto nivel, fácil de usar
 **TensorFlow Lite**: Para móviles
 **TensorFlow.js**: Para web

#### Instalación

```bash
pip install tensorflow
```

#### Ejemplo con Keras

```python
import tensorflow as tf
from tensorflow import keras

# Modelo simple
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Predecir
predictions = model.predict(x_test)
```

#### Cuándo Usar TensorFlow

- Deployment en producción a gran escala
- Aplicaciones móviles (TF Lite)
- Aplicaciones web (TF.js)
- Necesitas TensorFlow Serving

---

### 3. JAX

**Desarrollado por**: Google DeepMind

#### Características

 **Velocidad**: Extremadamente rápido
 **Funcional**: Programación funcional
 **Auto-diferenciación**: `grad()` function
 **JIT compilation**: Just-in-time compilation

#### Cuándo Considerar JAX

- Research avanzado
- Necesitas máxima velocidad
- Programación funcional
- Experimentos matemáticos

---

### PyTorch vs TensorFlow: Comparación

| Aspecto | PyTorch | TensorFlow |
|---------|---------|------------|
| **Curva de aprendizaje** | Más fácil | Más complejo |
| **Debugging** | Excelente | Bueno |
| **Research** | Preferido | Menos usado |
| **Producción** | Mejoró mucho | Mejor |
| **Comunidad NLP** | Dominante | Menor |
| **Móvil/Web** | Posible | Mejor soporte |

**Recomendación**:
- **Aprendiendo**: PyTorch (más intuitivo)
- **Producción enterprise**: TensorFlow
- **NLP/LLMs**: PyTorch + Hugging Face

---

## Hugging Face Ecosystem

**Hugging Face** es la plataforma líder para modelos de IA, especialmente NLP.

### 1. Transformers Library

La librería **más importante** para trabajar con LLMs.

#### Instalación

```bash
pip install transformers
```

#### Características

 **Miles de modelos preentrenados**
 **Uso extremadamente simple**
 **Integración con PyTorch y TensorFlow**
 **Pipelines para tareas comunes**

#### Ejemplo: Pipelines

```python
from transformers import pipeline

# Análisis de sentimientos
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Generación de texto
generator = pipeline("text-generation", model="gpt2")
text = generator("Once upon a time", max_length=50)

# Traducción
translator = pipeline("translation_en_to_es")
result = translator("Hello, how are you?")

# Resumen
summarizer = pipeline("summarization")
summary = summarizer(long_article, max_length=130)

# Question Answering
qa = pipeline("question-answering")
result = qa(question="What is AI?", context="AI is...")
```

#### Modelos Disponibles

**Generación de Texto**:
- GPT-2, GPT-Neo, GPT-J
- Llama 2, Mistral
- Falcon, Bloom

**Comprensión de Lenguaje**:
- BERT, RoBERTa
- ALBERT, DistilBERT
- DeBERTa

**Multilingüe**:
- mBERT, XLM-R
- mT5

#### Ejemplo: Usar Modelo Específico

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Cargar modelo y tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generar texto
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
generated = tokenizer.decode(outputs[0])
print(generated)
```

#### Fine-tuning Simplificado

```python
from transformers import Trainer, TrainingArguments

# Configurar entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
)

# Entrenar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

---

### 2. Datasets

Acceso a miles de datasets preparados.

```python
from datasets import load_dataset

# Cargar dataset
dataset = load_dataset("imdb")

# Ver estructura
print(dataset)

# Acceder a ejemplos
print(dataset['train'][0])

# Procesar
def preprocess(example):
    example['text'] = example['text'].lower()
    return example

dataset = dataset.map(preprocess)
```

---

### 3. Tokenizers

Tokenización ultrarrápida.

```python
from tokenizers import Tokenizer
from transformers import AutoTokenizer

# Usar tokenizer preentrenado
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenizar
encoded = tokenizer("Hello, world!")
print(encoded)
# {'input_ids': [101, 7592, 1010, 2088, 999, 102], 'attention_mask': [1, 1, 1, 1, 1, 1]}

# Decodificar
decoded = tokenizer.decode(encoded['input_ids'])
```

---

### 4. Accelerate

Simplifica entrenamiento distribuido.

```python
from accelerate import Accelerator

accelerator = Accelerator()

# Preparar modelo, optimizer, dataloader
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# Entrenar
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
```

---

### 5. Hugging Face Hub

**Repositorio de modelos y datasets**

```python
from huggingface_hub import login, HfApi

# Login
login(token="your_token")

# Subir modelo
model.push_to_hub("my-model-name")

# Descargar modelo
model = AutoModel.from_pretrained("username/model-name")
```

**Web**: [huggingface.co](https://huggingface.co)

---

## LangChain y Frameworks para LLMs

### 1. LangChain

**Framework para desarrollar aplicaciones con LLMs**

#### Instalación

```bash
pip install langchain openai
```

#### Conceptos Clave

**Chains**: Secuencias de operaciones
```python
from langchain import OpenAI, LLMChain, PromptTemplate

# Template
template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate(template=template, input_variables=["product"])

# LLM
llm = OpenAI(temperature=0.9)

# Chain
chain = LLMChain(llm=llm, prompt=prompt)

# Ejecutar
result = chain.run("colorful socks")
print(result)
```

**Agents**: Sistemas que toman decisiones
```python
from langchain.agents import load_tools, initialize_agent, AgentType

# Herramientas
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Agente
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Usar
agent.run("What is the population of France times 2?")
```

**Memory**: Mantener contexto
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi, I'm Alice")
conversation.predict(input="What's my name?")
# Respuesta: "Your name is Alice"
```

#### Casos de Uso

- Chatbots con contexto
- RAG (Retrieval-Augmented Generation)
- Agentes autónomos
- Análisis de documentos

---

### 2. LlamaIndex (GPT Index)

**Especializado en conectar LLMs con datos**

```python
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

# Cargar documentos
documents = SimpleDirectoryReader('data').load_data()

# Crear índice
index = GPTSimpleVectorIndex.from_documents(documents)

# Consultar
response = index.query("What did the author do growing up?")
print(response)
```

**Ideal para**:
- RAG sobre documentos propios
- Question answering sobre bases de conocimiento
- Búsqueda semántica

---

### 3. Semantic Kernel (Microsoft)

Framework de Microsoft para orquestar IA.

```python
import semantic_kernel as sk

kernel = sk.Kernel()

# Agregar servicio
kernel.add_text_completion_service(
    "gpt-3.5",
    OpenAITextCompletion("gpt-3.5-turbo", api_key)
)

# Crear función
summarize = kernel.create_semantic_function(
    "Summarize: {{$input}}"
)

result = summarize("Long text here...")
```

---

### 4. Haystack

**Framework de NLP para búsqueda y QA**

```python
from haystack.nodes import FARMReader, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline

# Retriever
retriever = BM25Retriever(document_store)

# Reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Pipeline
pipe = ExtractiveQAPipeline(reader, retriever)

# Consultar
result = pipe.run(
    query="What is the capital of France?",
    params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)
```

---

## APIs de Modelos Generativos

### 1. OpenAI API

#### Instalación

```bash
pip install openai
```

#### Uso Básico

```python
import openai

openai.api_key = "your-api-key"

# Chat Completion (GPT-3.5/4)
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

#### Generación de Imágenes (DALL-E)

```python
response = openai.Image.create(
    prompt="A cyberpunk cat",
    n=1,
    size="1024x1024"
)

image_url = response['data'][0]['url']
```

#### Embeddings

```python
response = openai.Embedding.create(
    input="Your text here",
    model="text-embedding-ada-002"
)

embeddings = response['data'][0]['embedding']
```

---

### 2. Anthropic API (Claude)

```bash
pip install anthropic
```

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

print(message.content)
```

---

### 3. Google AI (Gemini)

```bash
pip install google-generativeai
```

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("What is AI?")
print(response.text)
```

---

### 4. Cohere

```bash
pip install cohere
```

```python
import cohere

co = cohere.Client('your-api-key')

response = co.generate(
    prompt="Write a haiku about AI",
    max_tokens=50
)

print(response.generations[0].text)
```

---

## Herramientas y Utilidades

### 1. Jupyter Notebooks

Entorno interactivo para experimentación.

```bash
pip install jupyter
jupyter notebook
```

**Alternativas**:
- **Google Colab**: Gratis, GPU incluido
- **Kaggle Notebooks**: Datasets integrados
- **JupyterLab**: IDE más completo

---

### 2. Weights & Biases (W&B)

Tracking de experimentos.

```python
import wandb

wandb.init(project="my-project")

# Log metrics
wandb.log({"loss": 0.5, "accuracy": 0.9})

# Log modelo
wandb.save("model.pt")
```

---

### 3. MLflow

Gestión de ciclo de vida ML.

```python
import mlflow

mlflow.start_run()
mlflow.log_param("learning_rate", 0.01)
mlflow.log_metric("accuracy", 0.95)
mlflow.log_model(model, "model")
mlflow.end_run()
```

---

### 4. Gradio

Crear interfaces web para modelos.

```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```

---

### 5. Streamlit

Apps web para ML.

```python
import streamlit as st

st.title("My ML App")
user_input = st.text_input("Enter text:")

if st.button("Process"):
    result = process(user_input)
    st.write(result)
```

---

## <¯ Resumen del Módulo

En este módulo has aprendido:

 Python y sus librerías fundamentales para IA
 PyTorch y TensorFlow: frameworks de deep learning
 Hugging Face: ecosistema completo para LLMs
 LangChain y frameworks para aplicaciones con IA
 APIs de OpenAI, Anthropic, Google para usar modelos
 Herramientas complementarias para desarrollo

---

## =Ú Recursos Adicionales

### Documentación Oficial
- [PyTorch Docs](https://pytorch.org/docs/)
- [TensorFlow Docs](https://www.tensorflow.org/api_docs)
- [Hugging Face Docs](https://huggingface.co/docs)
- [LangChain Docs](https://python.langchain.com/)

### Cursos
- [Fast.ai](https://www.fast.ai/) - Deep Learning práctico
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Hugging Face Course](https://huggingface.co/learn)

---

##  Ejercicios Prácticos

1. Instala Python y crea un entorno virtual
2. Implementa una red neuronal simple en PyTorch
3. Usa un pipeline de Hugging Face para análisis de sentimientos
4. Crea una cadena simple con LangChain
5. Llama a la API de OpenAI para generar texto

---

## ¡ Próximo Módulo

Con las herramientas y librerías dominadas, ahora implementarás **proyectos prácticos** completos.

=I [Módulo 6: Ejemplos Prácticos](../modulo6/README.md)

[ Volver al Módulo 4](../modulo4/README.md) | [ Volver al inicio](../../README.md)
