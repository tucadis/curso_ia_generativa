# Módulo 4: Herramientas de IA Generativa

## =Ë Contenido del Módulo

1. [Modelos de Lenguaje (LLMs)](#modelos-de-lenguaje-llms)
2. [Generación de Imágenes](#generación-de-imágenes)
3. [Generación de Audio y Música](#generación-de-audio-y-música)
4. [Generación de Video](#generación-de-video)
5. [Herramientas para Desarrolladores](#herramientas-para-desarrolladores)
6. [Herramientas Especializadas](#herramientas-especializadas)

---

## Modelos de Lenguaje (LLMs)

### 1. ChatGPT (OpenAI)

**Modelo base**: GPT (Generative Pre-trained Transformer)

#### Versiones Disponibles

**GPT-3.5 Turbo**
- Rápido y eficiente
- Gratuito con limitaciones
- Contexto: 4K-16K tokens
- Ideal para tareas generales

**GPT-4**
- Razonamiento avanzado
- Mejor comprensión contextual
- Contexto: 8K-128K tokens
- Multimodal (texto + imágenes)
- Requiere suscripción ChatGPT Plus ($20/mes)

**GPT-4 Turbo**
- Más rápido que GPT-4
- Más económico (API)
- Contexto: 128K tokens
- Actualizado frecuentemente

#### Características Principales

 **Conversación natural**: Mantiene contexto largo
 **Multiuso**: Escritura, código, análisis, creatividad
 **Plugins**: Integración con herramientas externas
 **Código avanzado**: Excelente para programación
 **GPTs personalizados**: Crear asistentes especializados

#### Casos de Uso

- Escritura y edición de contenido
- Programación y debugging
- Análisis de datos
- Tutorías y educación
- Brainstorming e ideas
- Traducción
- Resúmenes de documentos

#### Limitaciones

L Datos hasta abril 2023 (GPT-4)
L Puede "alucinar" información
L No tiene acceso a internet (sin plugins)
L Límites de uso en plan gratuito

#### Precios

- **ChatGPT Free**: GPT-3.5, limitado
- **ChatGPT Plus**: $20/mes, GPT-4, acceso prioritario
- **API**: Pay-per-use, desde $0.0015/1K tokens

---

### 2. Claude (Anthropic)

**Desarrollado por**: Anthropic (fundada por ex-OpenAI)

#### Versiones Disponibles

**Claude 3 Haiku**
- Más rápido y económico
- Tareas simples y rutinarias
- Contexto: 200K tokens

**Claude 3 Sonnet**
- Balance rendimiento/costo
- Uso general
- Contexto: 200K tokens

**Claude 3 Opus**
- Más capaz y sofisticado
- Tareas complejas
- Contexto: 200K tokens
- Multimodal

#### Características Destacadas

 **Contexto largo**: 200K tokens (~150,000 palabras)
 **Seguridad**: Enfoque en "Constitutional AI"
 **Honestidad**: Admite cuando no sabe algo
 **Análisis de documentos**: Excelente con PDFs, código
 **Multimodal**: Análisis de imágenes (Opus)

#### Ventajas sobre GPT

- Contexto mucho más largo
- Menos propenso a alucinaciones
- Mejor con documentos largos
- Más cuidadoso con información sensible

#### Casos de Uso Ideales

- Análisis de documentos largos
- Research y síntesis
- Tareas que requieren precisión
- Código complejo
- Edición de contenido extenso

#### Acceso

- **claude.ai**: Interfaz web (gratis y Pro)
- **Claude Pro**: $20/mes
- **API**: Disponible para desarrolladores

---

### 3. Gemini (Google)

**Anteriormente**: Bard

#### Versiones

**Gemini 1.0 Pro**
- Uso general
- Integrado con Google

**Gemini 1.5 Pro**
- Contexto: 1 millón de tokens
- Multimodal nativo
- Razonamiento mejorado

**Gemini Ultra**
- Más capaz
- Próximo lanzamiento comercial

#### Características Únicas

 **Integración Google**: Búsqueda, Gmail, Docs, Maps
 **Contexto masivo**: Hasta 1M tokens (1.5 Pro)
 **Multimodal nativo**: Texto, imágenes, audio, video
 **Acceso a información actual**: Via Google Search
 **Gratuito**: Con cuenta Google

#### Ventajas

- Información actualizada
- Integración con ecosistema Google
- Contexto extremadamente largo
- Multimodal desde el diseño

#### Casos de Uso

- Research con información actual
- Análisis de videos y audio
- Integración con Google Workspace
- Procesamiento de documentos masivos

#### Acceso

- **gemini.google.com**: Gratis
- **Gemini Advanced**: $20/mes (parte de Google One AI Premium)
- **API**: Google AI Studio

---

### 4. LLaMA (Meta)

**Características**: Open source

#### Versiones

**LLaMA 2**
- 7B, 13B, 70B parámetros
- Código abierto
- Uso comercial permitido

**LLaMA 3**
- Mejoras significativas
- Más capaz que LLaMA 2
- Lanzamiento reciente

#### Ventajas

 **Open source**: Código y pesos disponibles
 **Privacidad**: Ejecutar localmente
 **Personalizable**: Fine-tuning completo
 **Gratis**: Sin costos de API

#### Desafíos

L Requiere hardware potente
L Setup técnico complejo
L Sin interfaz amigable por defecto

#### Casos de Uso

- Desarrollo de productos
- Fine-tuning especializado
- Research
- Aplicaciones con requisitos de privacidad

---

### 5. Otras Herramientas de Lenguaje

#### Perplexity AI
- **Especialidad**: Búsqueda conversacional con fuentes
- **Ventaja**: Cita fuentes, información actual
- **Ideal para**: Research, fact-checking

#### Claude Code (antes Claude Developer)
- **Especialidad**: Programación asistida
- **Integración**: VSCode, IDEs
- **Ideal para**: Desarrollo de software

#### Mistral AI
- **Origen**: Francia (Europa)
- **Modelos**: Mixtral, Mistral 7B
- **Ventaja**: Open source, eficiente

#### Cohere
- **Enfoque**: APIs empresariales
- **Características**: Embeddings, clasificación, generación

---

## Generación de Imágenes

### 1. Midjourney

#### Características

 **Calidad artística excepcional**
 **Estilos cinematográficos**
 **Comunidad activa**
 **Actualizaciones constantes**

#### Versiones

- **V5**: Fotorealismo mejorado
- **V6**: Comprensión de prompts mejorada
- **Niji**: Especializado en anime

#### Acceso

- **Plataforma**: Discord
- **Precios**:
  - Basic: $10/mes (200 imágenes)
  - Standard: $30/mes (ilimitado relax)
  - Pro: $60/mes (modo stealth)

#### Prompts en Midjourney

```
/imagine prompt: a cyberpunk cat warrior, neon lights,
highly detailed, 8k, cinematic lighting --ar 16:9 --v 6
```

**Parámetros comunes:**
- `--ar`: Aspect ratio (ej: 16:9, 1:1)
- `--v`: Versión (5, 6)
- `--style`: Estilo específico
- `--chaos`: Variabilidad (0-100)

#### Mejor para

- Arte conceptual
- Diseño de personajes
- Escenas cinematográficas
- Ilustraciones fantásticas

---

### 2. DALL-E 3 (OpenAI)

#### Características

 **Integrado con ChatGPT**
 **Comprensión avanzada de prompts**
 **Seguro y moderado**
 **Coherencia en detalles**

#### Capacidades

- **Generación desde texto**
- **Edición (inpainting)**
- **Variaciones**
- **Diferentes estilos**

#### Acceso

- **ChatGPT Plus**: Incluido
- **API**: Disponible para desarrolladores
- **Bing Image Creator**: Gratis (limitado)

#### Ventajas

- Más fiel al prompt
- Menos alucinaciones visuales
- Seguridad incorporada
- Fácil de usar

---

### 3. Stable Diffusion

#### Características Clave

 **100% Open Source**
 **Ejecutable localmente**
 **Altamente personalizable**
 **Comunidad masiva**
 **Sin censura (a elección)**

#### Versiones

- **SD 1.5**: Estándar de la industria
- **SD 2.0/2.1**: Mejoras técnicas
- **SDXL**: Calidad superior, más parámetros

#### Herramientas Populares

**Automatic1111 WebUI**
- Interfaz web local
- Extensiones abundantes
- Control fino

**ComfyUI**
- Basado en nodos
- Workflows complejos
- Más técnico

**DreamStudio**
- Oficial de Stability AI
- Cloud-based
- Fácil de usar

#### Extensiones Importantes

**ControlNet**
- Control preciso de composición
- Pose, profundidad, bordes
- Esencial para control profesional

**LoRA (Low-Rank Adaptation)**
- Fine-tuning ligero
- Estilos personalizados
- Personajes específicos

#### Acceso

- **Gratis**: Instalar localmente
- **Replicate**: Pay-per-use en cloud
- **DreamStudio**: Créditos

#### Requisitos Locales

- GPU: NVIDIA con 6GB+ VRAM
- RAM: 16GB+
- Espacio: 20GB+

---

### 4. Leonardo.ai

#### Características

- Enfocado en game assets
- Control fino de generación
- Modelos especializados
- Canvas de edición

#### Ideal para

- Diseño de videojuegos
- Texturas
- Concept art
- Assets comerciales

---

### 5. Adobe Firefly

#### Características

 **Integrado en Adobe Suite**
 **Entrenado con Adobe Stock**
 **Comercialmente seguro**
 **Generative Fill en Photoshop**

#### Ventajas

- Sin problemas de copyright
- Integración profesional
- Flujo de trabajo establecido

#### Acceso

- Adobe Creative Cloud
- Standalone beta

---

## Generación de Audio y Música

### 1. Eleven Labs

**Especialidad**: Clonación de voz y text-to-speech

#### Características

 **Voces ultra realistas**
 **Clonación de voz**
 **Múltiples idiomas**
 **Control emocional**

#### Casos de Uso

- Narración de audiolibros
- Doblaje
- Podcasts
- Asistentes virtuales

#### Precios

- Free: 10,000 caracteres/mes
- Starter: $5/mes
- Creator: $22/mes

---

### 2. Suno AI

**Especialidad**: Generación de música completa

#### Características

 **Crea canciones completas**
 **Voz, letra, instrumentación**
 **Diversos géneros**
 **Alta calidad**

#### Acceso

- suno.ai
- Planes gratuitos y premium

---

### 3. MusicLM (Google)

- Generación de música desde texto
- Experimental
- Alta calidad

---

### 4. Mubert

- Generación de música para contenido
- Royalty-free
- Ideal para streamers, creadores

---

## Generación de Video

### 1. Runway Gen-2

#### Características

 **Text-to-Video**
 **Image-to-Video**
 **Video editing con IA**
 **Herramientas profesionales**

#### Capacidades

- Generación de clips cortos
- Edición inteligente
- Efectos visuales
- Remove backgrounds

---

### 2. Pika Labs

- Generación de video desde texto
- Animación de imágenes
- Fácil de usar

---

### 3. Sora (OpenAI)

**Estado**: Limitado, no público aún

#### Características Anunciadas

- Videos de hasta 60 segundos
- Alta calidad y coherencia
- Text-to-video avanzado
- Comprensión física del mundo

---

### 4. Synthesia

**Especialidad**: Avatares hablantes

#### Casos de Uso

- Videos corporativos
- Educación
- Presentaciones
- Localización multiidioma

---

## Herramientas para Desarrolladores

### 1. GitHub Copilot

**Desarrollado por**: GitHub + OpenAI

#### Características

 **Autocompletado inteligente**
 **Generación de funciones completas**
 **Sugerencias contextuales**
 **Múltiples lenguajes**

#### Integración

- VSCode
- JetBrains IDEs
- Neovim
- Visual Studio

#### Capacidades

- Completar código
- Generar tests
- Escribir documentación
- Refactorizar código

#### Precio

- $10/mes individual
- $19/mes negocios
- Gratis para estudiantes y open source

---

### 2. Cursor

**Concepto**: IDE con IA nativa

#### Características

 **Chat con tu codebase**
 **Edición multi-archivo**
 **Comandos de voz**
 **Based on VSCode**

#### Ventajas

- Comprende todo el proyecto
- Ediciones inteligentes
- Debugging asistido

---

### 3. Replit Ghostwriter

- IDE en la nube
- IA integrada
- Deployment fácil

---

### 4. Tabnine

- Alternativa a Copilot
- Privacidad-focused
- On-premise disponible

---

### 5. Amazon CodeWhisperer

- Gratuito
- Integración AWS
- Seguridad incorporada

---

## Herramientas Especializadas

### Productividad

**Notion AI**
- Escritura asistida
- Resúmenes
- Brainstorming

**Grammarly**
- Corrección gramática
- Estilo y tono
- Reescritura

**Jasper**
- Marketing copy
- Contenido para redes
- Blogs

### Diseño

**Canva AI**
- Magic Write
- Background Remover
- Image Generator

**Figma AI**
- Diseño asistido
- Prototipos

### Presentaciones

**Gamma**
- Generación de presentaciones
- Desde texto a slides

**Beautiful.ai**
- Diseño automático
- Templates inteligentes

### Research

**Consensus**
- Búsqueda en papers científicos
- Resúmenes de investigación

**Elicit**
- Asistente de investigación
- Análisis de literatura

### Transcripción

**Otter.ai**
- Transcripción en tiempo real
- Resúmenes de reuniones

**Whisper (OpenAI)**
- Transcripción precisa
- Múltiples idiomas
- Open source

---

## Comparativa Rápida

### Modelos de Lenguaje

| Herramienta | Mejor para | Contexto | Precio |
|-------------|-----------|----------|--------|
| **ChatGPT** | Uso general, código | 4K-128K | $0-20/mes |
| **Claude** | Documentos largos | 200K | $0-20/mes |
| **Gemini** | Investigación actual | 1M | Gratis |
| **LLaMA** | Privacidad, custom | Variable | Gratis |

### Generación de Imágenes

| Herramienta | Mejor para | Acceso | Precio |
|-------------|-----------|--------|--------|
| **Midjourney** | Arte, cinematográfico | Discord | $10-60/mes |
| **DALL-E 3** | Fidelidad al prompt | ChatGPT | $20/mes |
| **Stable Diffusion** | Control total, gratis | Local/Cloud | Gratis |
| **Firefly** | Comercial, Adobe | Adobe CC | Incluido |

---

## <¯ Resumen del Módulo

En este módulo has aprendido:

 Principales modelos de lenguaje y sus diferencias
 Herramientas de generación de imágenes y cuándo usar cada una
 Opciones para generar audio, música y video
 Herramientas esenciales para desarrolladores
 Herramientas especializadas por industria
 Comparativas para tomar decisiones informadas

---

## =Ú Recursos Adicionales

### Directorios de Herramientas
- [There's An AI For That](https://theresanaiforthat.com/)
- [Futurepedia](https://www.futurepedia.io/)
- [AI Tools Directory](https://aitoolsdirectory.com/)

### Comunidades
- Reddit: r/ArtificialIntelligence, r/StableDiffusion
- Discord: Midjourney, Stable Diffusion
- Twitter: Sigue hashtags #AItools #GenerativeAI

---

##  Ejercicios Prácticos

1. Prueba al menos 3 LLMs diferentes con el mismo prompt
2. Genera imágenes con 2+ herramientas y compara resultados
3. Experimenta con GitHub Copilot o alternativa
4. Crea un audio con Eleven Labs
5. Explora una herramienta de cada categoría

---

## ¡ Próximo Módulo

Ahora que conoces las herramientas, aprenderás los **lenguajes y librerías** para implementar IA Generativa programáticamente.

=I [Módulo 5: Lenguajes y Librerías](../modulo5/README.md)

[ Volver al Módulo 3](../modulo3/README.md) | [ Volver al inicio](../../README.md)
