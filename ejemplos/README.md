# Ejemplos Prácticos de Código

Esta carpeta contiene ejemplos de código ejecutables de los proyectos del Módulo 6.

## =Ë Contenido

- `requirements.txt` - Dependencias necesarias
- `ejemplo_generador_texto.py` - Generación de texto con OpenAI API
- `ejemplo_chatbot.py` - Chatbot con memoria
- `ejemplo_rag.py` - Sistema RAG básico
- `ejemplo_streamlit_app.py` - Aplicación web con IA

## =€ Instalación

1. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno:
```bash
# Crear archivo .env
echo "OPENAI_API_KEY=tu-api-key-aqui" > .env
```

## ¶ Ejecutar Ejemplos

### Generador de Texto
```bash
python ejemplo_generador_texto.py
```

### Chatbot
```bash
python ejemplo_chatbot.py
```

### Sistema RAG
```bash
# Primero coloca tus documentos en ./mis_documentos/
python ejemplo_rag.py
```

### Aplicación Web
```bash
streamlit run ejemplo_streamlit_app.py
```

## =Ý Notas Importantes

- Necesitas una API key de OpenAI para los ejemplos
- Algunos ejemplos requieren GPU para mejor rendimiento (opcional)
- Los ejemplos son educativos y deben adaptarse para producción

## > Contribuciones

Si encuentras errores o quieres agregar ejemplos, ¡las contribuciones son bienvenidas!

## =Ú Recursos

- [Documentación OpenAI](https://platform.openai.com/docs)
- [Documentación Streamlit](https://docs.streamlit.io)
- [Documentación LangChain](https://python.langchain.com/)
