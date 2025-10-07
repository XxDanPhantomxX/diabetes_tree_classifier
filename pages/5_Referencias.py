import streamlit as st
st.title("Conclusión y Referencias")
st.write("## Conclusión")
st.write("""
- El modelo de Árbol de Decisión es efectivo para predecir la diabetes basándose en síntomas específicos.
- La selección adecuada de características es crucial para mejorar la precisión del modelo.
- La interpretación del modelo es sencilla, lo que facilita su uso en entornos clínicos.
- La aplicación interactiva permite a los usuarios ingresar síntomas y obtener predicciones rápidas.
""")
st.write("## Referencias")
st.write("""
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html): Documentación oficial de la biblioteca Scikit-learn utilizada para construir y evaluar el modelo de Árbol de Decisión.
- [Streamlit Documentation](https://docs.streamlit.io/): Documentación oficial de Streamlit, la biblioteca utilizada para crear la aplicación web interactiva.
- [Pandas Documentation](https://pandas.pydata.org/docs/): Documentación oficial de Pandas, la biblioteca utilizada para la manipulación y análisis de datos.
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html): Documentación oficial de Matplotlib, la biblioteca utilizada para la visualización de datos.
- [Artículo de Arjun, R. S., & Vigneshwaran, M](https://doi.org/10.37418/amsj.9.2): Un artículo que explica la teoría de conjunto rugoso y su aplicación en la clasificación con los 25 pacientes.
""")
st.info("Esta aplicación es solo para fines educativos y no debe utilizarse como un diagnóstico médico real. Consulta siempre a un profesional de la salud para obtener asesoramiento médico adecuado.")
st.success("¡Gracias por visitar la aplicación!")
