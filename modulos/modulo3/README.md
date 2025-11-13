# Módulo 3: Introducción a IA Generativa

## =Ë Contenido del Módulo

1. [¿Qué es la IA Generativa?](#qué-es-la-ia-generativa)
2. [Arquitecturas Principales](#arquitecturas-principales)
3. [Large Language Models (LLMs)](#large-language-models-llms)
4. [Modelos de Generación de Imágenes](#modelos-de-generación-de-imágenes)
5. [Modelos Multimodales](#modelos-multimodales)
6. [Conceptos Técnicos Fundamentales](#conceptos-técnicos-fundamentales)

---

## ¿Qué es la IA Generativa?

La **IA Generativa** es un tipo de inteligencia artificial capaz de **crear contenido nuevo y original** que no existía previamente, incluyendo texto, imágenes, audio, video, código y más.

### Definición

> "La IA Generativa es un conjunto de algoritmos de aprendizaje automático que pueden generar nuevos datos similares a los datos con los que fueron entrenados."

### Diferencia Clave con IA Tradicional

**IA Tradicional (Discriminativa):**
```
Pregunta: "¿Qué hay en esta imagen?"
Respuesta: "Un gato"
```

**IA Generativa:**
```
Instrucción: "Crea una imagen de un gato astronauta en Marte"
Resultado: [Genera una imagen completamente nueva]
```

### Características Principales

 **Creativa**: Genera contenido original
 **Versátil**: Múltiples modalidades (texto, imagen, audio, video)
 **Contextual**: Comprende y mantiene contexto
 **Adaptable**: Se ajusta a diferentes estilos y tonos
 **Interactiva**: Permite conversaciones naturales

### ¿Cómo Funciona?

La IA Generativa funciona mediante:

1. **Entrenamiento con datos masivos**
   - Aprende patrones de millones o billones de ejemplos
   - Modelos de lenguaje: libros, artículos, sitios web
   - Modelos de imagen: millones de imágenes etiquetadas

2. **Compresión de conocimiento**
   - Captura la "esencia" de los datos
   - Aprende relaciones y patrones complejos

3. **Generación probabilística**
   - Predice qué viene después
   - Muestrea de distribuciones de probabilidad
   - Introduce creatividad mediante aleatoriedad controlada

---

## Arquitecturas Principales

### 1. GANs - Generative Adversarial Networks

**Creado por**: Ian Goodfellow (2014)

#### Concepto

Dos redes neuronales compiten entre sí:
- **Generador**: Crea datos falsos
- **Discriminador**: Distingue entre datos reales y falsos

```
Generador ’ Imagen Falsa 
                         ’ Discriminador ’ Real o Falso
Datos Reales ’ Imagen Real 
```

#### Proceso de Entrenamiento

1. **Generador** crea una imagen falsa
2. **Discriminador** intenta detectar si es real o falsa
3. **Generador** mejora para engañar al discriminador
4. **Discriminador** mejora para detectar falsificaciones
5. Repetir hasta alcanzar equilibrio

#### Analogía

Como un **falsificador** (generador) y un **detective** (discriminador):
- El falsificador intenta crear billetes falsos
- El detective aprende a detectarlos
- El falsificador mejora sus técnicas
- El detective mejora su detección
- Eventualmente, los billetes falsos son indistinguibles de los reales

#### Ventajas
 Genera imágenes de alta calidad
 Útil para aumentar datasets
 Aplicaciones en arte y diseño

#### Desafíos
L Difícil de entrenar (inestabilidad)
L Puede colapsar (genera siempre lo mismo)
L Requiere ajuste cuidadoso de hiperparámetros

#### Aplicaciones
- **StyleGAN**: Generación de rostros realistas
- **CycleGAN**: Transferencia de estilo (foto ’ pintura)
- **Pix2Pix**: Traducción imagen a imagen
- **DeepFake**: Síntesis de rostros (controvertido)

---

### 2. VAEs - Variational Autoencoders

**Concepto**: Codificadores automáticos variaciones

#### Arquitectura

```
Entrada ’ Encoder ’ Espacio Latente ’ Decoder ’ Salida
```

#### Componentes

1. **Encoder (Codificador)**
   - Comprime la entrada en representación compacta
   - Captura características esenciales

2. **Espacio Latente**
   - Representación comprimida de los datos
   - Espacio continuo y estructurado

3. **Decoder (Decodificador)**
   - Reconstruye datos desde el espacio latente
   - Genera nuevas variaciones

#### Cómo Funciona

1. Encoder comprime imagen ’ vector latente
2. Agregar variación aleatoria al vector
3. Decoder reconstruye desde vector ’ nueva imagen

#### Ventajas
 Entrenamiento más estable que GANs
 Genera variaciones controladas
 Interpolación suave entre conceptos

#### Aplicaciones
- Generación de rostros
- Compresión de imágenes
- Detección de anomalías
- Generación de moléculas (farmacéutica)

---

### 3. Transformers

**Paper seminal**: "Attention Is All You Need" (2017)

#### Revolución en IA

Los Transformers **revolucionaron** el procesamiento de lenguaje natural y se convirtieron en la base de:
- GPT (ChatGPT)
- BERT
- Claude
- Gemini
- LLaMA
- Y prácticamente todos los LLMs modernos

#### Concepto Clave: Atención (Attention)

**Mecanismo de Atención**: Permite al modelo enfocarse en las partes relevantes de la entrada.

**Ejemplo:**
```
Frase: "El gato comió su comida porque tenía hambre"
Pregunta: ¿Qué significa "tenía"?

Atención: El modelo se enfoca en "El gato" para entender que
"tenía" se refiere al gato, no a la comida.
```

#### Arquitectura

```
Entrada (tokens)
    “
Embeddings (representaciones vectoriales)
    “
Multi-Head Attention (múltiples cabezales de atención)
    “
Feed-Forward Network
    “
Salida
```

#### Componentes Clave

1. **Self-Attention**: Cada palabra "mira" a todas las demás
2. **Multi-Head Attention**: Múltiples perspectivas simultáneas
3. **Positional Encoding**: Información de orden/posición
4. **Feed-Forward Networks**: Procesamiento adicional

#### Ventajas sobre RNNs/LSTMs
 **Paralelización**: Procesa toda la secuencia simultáneamente
 **Contexto largo**: Captura dependencias a larga distancia
 **Escalabilidad**: Funciona mejor con más datos y parámetros
 **Transferencia**: Preentrenamiento + fine-tuning

#### Variantes

**Encoder-only** (BERT):
- Comprensión de lenguaje
- Clasificación
- Análisis de sentimientos

**Decoder-only** (GPT):
- Generación de texto
- Conversación
- Completación

**Encoder-Decoder** (T5):
- Traducción
- Resumen
- Pregunta-respuesta

---

### 4. Diffusion Models (Modelos de Difusión)

**Concepto**: Base de Stable Diffusion, DALL-E 2, Midjourney

#### Cómo Funciona

**Proceso de Difusión (Forward)**:
```
Imagen Clara ’ + Ruido ’ + Más Ruido ’ ... ’ Ruido Puro
```

**Proceso de Eliminación de Ruido (Reverse)**:
```
Ruido Puro ’ - Ruido ’ - Más Ruido ’ ... ’ Imagen Clara
```

#### Entrenamiento

1. Tomar imagen real
2. Agregar ruido gradualmente
3. Entrenar modelo para **predecir y eliminar** el ruido
4. Repetir millones de veces

#### Generación

1. Comenzar con ruido aleatorio
2. Guiar con texto: "un gato astronauta"
3. Aplicar modelo paso a paso para eliminar ruido
4. Resultado: imagen guiada por el texto

#### Ventajas
 Calidad excepcional de imágenes
 Control fino sobre generación
 Estable y predecible
 Edición y manipulación flexible

#### Modelos Famosos
- **Stable Diffusion**: Open source
- **DALL-E 2/3**: OpenAI
- **Midjourney**: Imágenes artísticas
- **Imagen**: Google

---

## Large Language Models (LLMs)

Los **Modelos de Lenguaje Grandes** son el tipo más popular de IA Generativa actualmente.

### ¿Qué son los LLMs?

Modelos de aprendizaje profundo entrenados con **billones de palabras** de texto para:
- Comprender lenguaje natural
- Generar texto coherente
- Razonar y resolver problemas
- Seguir instrucciones
- Mantener conversaciones

### Arquitectura Base: Transformers

Todos los LLMs modernos usan arquitectura Transformer:
- **GPT**: Generative Pre-trained Transformer
- **BERT**: Bidirectional Encoder Representations from Transformers
- **T5**: Text-to-Text Transfer Transformer

### Evolución de los LLMs

#### Era Pre-Transformer (antes de 2017)
- RNNs, LSTMs
- Limitados en contexto
- Difíciles de entrenar

#### Era Transformer (2017-2019)
- **BERT** (2018): Revoluciona NLP
- Preentrenamiento + Fine-tuning
- Transfer learning masivo

#### Era de Scaling (2020-2022)
- **GPT-3** (2020): 175 mil millones de parámetros
- Demostración de "emergent abilities"
- Few-shot learning

#### Era de Alignment (2022-presente)
- **ChatGPT** (Nov 2022): IA conversacional masiva
- **GPT-4** (2023): Multimodal
- **Claude, Gemini**: Competencia intensa
- Enfoque en seguridad y alineación

### Conceptos Clave en LLMs

#### 1. Parámetros

Número de valores aprendidos en el modelo.

**Escala:**
- GPT-2: 1.5 mil millones
- GPT-3: 175 mil millones
- GPT-4: ~1.7 billones (estimado)
- Claude 3: No revelado
- LLaMA 2: 7B, 13B, 70B variantes

**Más parámetros ` siempre mejor**
- También importa calidad de datos
- Arquitectura
- Entrenamiento

#### 2. Tokens

Unidades de texto procesadas por el modelo.

**Ejemplos:**
```
"Inteligencia" ’ ["Intel", "igencia"] (2 tokens)
"AI" ’ ["AI"] (1 token)
"ChatGPT es increíble" ’ ~4-5 tokens
```

**Límites de contexto:**
- GPT-3.5: 4,096 tokens (~3,000 palabras)
- GPT-4: 8,192 o 32,768 tokens
- Claude 3: 200,000 tokens
- Gemini 1.5 Pro: 1,000,000 tokens

#### 3. Temperatura

Controla la aleatoriedad/creatividad de las respuestas.

```
Temperatura 0.0 ’ Determinista, repetible
Temperatura 0.7 ’ Balanceado (default común)
Temperatura 1.0+ ’ Creativo, impredecible
```

**Cuándo usar:**
- **Baja (0-0.3)**: Código, matemáticas, respuestas factuales
- **Media (0.5-0.8)**: Escritura general, conversación
- **Alta (0.9-1.5)**: Creatividad, brainstorming, ficción

#### 4. Prompts

Instrucciones dadas al modelo.

**Técnicas de prompting:**

**Zero-shot**: Sin ejemplos
```
"Traduce al francés: Hello world"
```

**Few-shot**: Con ejemplos
```
"Traduce al francés:
Hello ’ Bonjour
Goodbye ’ Au revoir
Thank you ’ ?"
```

**Chain-of-Thought**: Razonamiento paso a paso
```
"Resuelve paso a paso: Si Juan tiene 3 manzanas y compra 2 más..."
```

#### 5. Fine-tuning

Ajustar un modelo preentrenado para una tarea específica.

**Proceso:**
1. Comenzar con modelo base (GPT, LLaMA)
2. Entrenar con datos específicos
3. Optimizar para tarea particular

**Tipos:**
- **Supervised Fine-tuning**: Con datos etiquetados
- **RLHF** (Reinforcement Learning from Human Feedback): ChatGPT usa esto
- **LoRA**: Fine-tuning eficiente

---

## Modelos de Generación de Imágenes

### Evolución

#### Fase 1: GANs (2014-2020)
- StyleGAN, BigGAN
- Alta calidad pero difícil control

#### Fase 2: DALL-E (2021)
- OpenAI combina lenguaje + imágenes
- Generación de imágenes desde texto

#### Fase 3: Diffusion Models (2022-presente)
- Stable Diffusion, Midjourney, DALL-E 2
- Calidad excepcional
- Control fino

### Modelos Principales

#### 1. DALL-E 2/3 (OpenAI)

**Características:**
- Generación desde texto detallado
- Edición (inpainting, outpainting)
- Variaciones de imágenes

**Capacidades:**
- Comprende conceptos complejos
- Combina ideas abstractas
- Estilos artísticos diversos

#### 2. Midjourney

**Características:**
- Enfocado en arte y estética
- Resultados cinematográficos
- Comunidad activa en Discord

**Fortalezas:**
- Calidad artística superior
- Estilos únicos
- Composición visual excelente

#### 3. Stable Diffusion

**Características:**
- **Open source**
- Puede ejecutarse localmente
- Altamente personalizable

**Ventajas:**
- Gratuito
- Control total
- Fine-tuning posible
- Plugins y extensiones

#### 4. Adobe Firefly

**Características:**
- Integrado en Adobe Suite
- Entrenado con datos licenciados
- Seguro comercialmente

### Conceptos en Generación de Imágenes

#### Text-to-Image (Texto a Imagen)
```
Prompt: "Un dragón steampunk volando sobre Nueva York al atardecer"
’ Modelo genera la imagen
```

#### Image-to-Image (Imagen a Imagen)
```
Imagen base + Prompt: "Convertir en estilo acuarela"
’ Transforma la imagen manteniendo estructura
```

#### Inpainting (Relleno)
```
Imagen + Área seleccionada + Prompt: "Agregar un gato aquí"
’ Rellena el área seleccionada
```

#### Outpainting (Extensión)
```
Imagen + Extender bordes
’ Genera contenido más allá de los límites originales
```

#### ControlNet

Técnica para **control preciso** de generación:
- Control de pose
- Control de profundidad
- Control de bordes
- Control de color

---

## Modelos Multimodales

Modelos que trabajan con **múltiples tipos de datos** simultáneamente.

### GPT-4 Vision

**Capacidades:**
- Analiza imágenes
- Responde preguntas sobre imágenes
- Lee gráficos y diagramas
- Genera texto desde imágenes

**Ejemplo:**
```
[Imagen de un gráfico]
Usuario: "Explica esta gráfica"
GPT-4: Analiza y explica tendencias, datos, insights
```

### Gemini (Google)

**Multimodal nativo:**
- Texto
- Imágenes
- Audio
- Video

**Ventaja:** Entrenado multimodalmente desde el inicio.

### Claude 3 (Anthropic)

**Capacidades:**
- Análisis de imágenes
- Documentos complejos
- Gráficos y diagramas
- Contexto muy largo (200K tokens)

### DALL-E 3 + ChatGPT

**Integración:**
- Conversar sobre imágenes
- Generar basándose en conversación
- Iterar diseños

---

## Conceptos Técnicos Fundamentales

### 1. Embeddings (Incrustaciones)

Representaciones vectoriales de datos.

**Ejemplo:**
```
"gato" ’ [0.2, -0.5, 0.8, ..., 0.1] (vector de 1536 dimensiones)
"perro" ’ [0.3, -0.4, 0.7, ..., 0.2]
```

**Propiedad importante:**
- Palabras similares ’ vectores cercanos
- Permite matemática con conceptos:
  - Rey - Hombre + Mujer H Reina

### 2. Tokenización

Proceso de convertir texto en tokens.

**Ejemplo:**
```
Texto: "ChatGPT es increíble"
Tokens: ["Chat", "GPT", " es", " incre", "íble"]
IDs: [1234, 5678, 901, 2345, 6789]
```

### 3. Attention Mechanisms

Ya discutido en Transformers, pero clave para:
- Enfocarse en partes relevantes
- Capturar contexto
- Comprender relaciones

### 4. Preentrenamiento

Entrenamiento inicial con enormes datasets.

**Proceso:**
1. **Unsupervised learning** con billones de palabras
2. Aprende patrones generales del lenguaje
3. Crea base de conocimiento amplio

### 5. Fine-tuning y Alignment

**Fine-tuning:**
- Especializar para tareas específicas

**Alignment (Alineación):**
- RLHF: Reinforcement Learning from Human Feedback
- Hacer que el modelo sea útil, honesto, inofensivo

**Proceso ChatGPT:**
1. Preentrenamiento (GPT base)
2. Supervised Fine-tuning
3. Reward Model Training
4. PPO (Proximal Policy Optimization)

### 6. Emergent Abilities

Capacidades que **emergen** a cierta escala que no existían en modelos pequeños:
- Razonamiento complejo
- Few-shot learning
- Instrucción-following
- Chain-of-thought

### 7. Hallucinations (Alucinaciones)

Cuando el modelo **genera información falsa** con confianza.

**Por qué ocurre:**
- Modelo predice patrones, no "conoce" verdad
- Rellena vacíos de conocimiento
- Optimizado para coherencia, no exactitud

**Mitigaciones:**
- RAG (Retrieval-Augmented Generation)
- Verificación de fuentes
- Temperature baja para tareas factuales

---

## <¯ Resumen del Módulo

En este módulo has aprendido:

 Qué es IA Generativa y cómo difiere de IA tradicional
 Arquitecturas principales: GANs, VAEs, Transformers, Diffusion Models
 LLMs: funcionamiento, conceptos clave, evolución
 Modelos de generación de imágenes y sus capacidades
 Modelos multimodales y su potencial
 Conceptos técnicos fundamentales para entender IA Generativa

---

## =Ú Recursos Adicionales

### Papers Fundamentales
- "Attention Is All You Need" (Transformers)
- "Generative Adversarial Networks" (GANs)
- "Denoising Diffusion Probabilistic Models"
- "Language Models are Few-Shot Learners" (GPT-3)

### Cursos
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [Fast.ai Stable Diffusion](https://www.fast.ai/)
- [DeepLearning.AI Short Courses](https://www.deeplearning.ai/short-courses/)

### Blogs Técnicos
- [OpenAI Blog](https://openai.com/blog)
- [Anthropic Research](https://www.anthropic.com/research)
- [Google AI Blog](https://ai.googleblog.com/)
- [Hugging Face Blog](https://huggingface.co/blog)

---

##  Ejercicios Prácticos

1. Experimenta con diferentes prompts en ChatGPT o Claude
2. Prueba generar imágenes con diferentes estilos en Midjourney/DALL-E
3. Compara respuestas con diferentes temperaturas
4. Investiga un paper de los mencionados
5. Identifica casos donde un modelo "alucina" información

---

## ¡ Próximo Módulo

Ahora que entiendes cómo funciona la IA Generativa, es momento de conocer las **herramientas** disponibles y cómo usarlas.

=I [Módulo 4: Herramientas de IA Generativa](../modulo4/README.md)

[ Volver al Módulo 2](../modulo2/README.md) | [ Volver al inicio](../../README.md)
