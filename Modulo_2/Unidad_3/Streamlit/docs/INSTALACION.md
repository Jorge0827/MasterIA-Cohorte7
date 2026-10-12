# Instalación — Clase Streamlit (carpeta aislada)

Mismo entorno `py_env`. No hace falta Java, Spark ni PostgreSQL.

Los `.joblib` **ya están** en `streamlit/modelos/`. No entrenes para la demo
salvo que falten.

```powershell
conda activate py_env
cd C:\Programacion\DataScienceIA\Docente_Dev_Senior_code\files\persisitencia\streamlit

pip install -r requirements.txt
```

Comprobar:

```powershell
python -c "import streamlit, sklearn, joblib; print(streamlit.__version__, sklearn.__version__)"
```

Arrancar las apps (deja la terminal abierta):

```powershell
streamlit run app_hola.py
streamlit run app.py
```

El navegador abre `http://localhost:8501`.

Si falta un `.joblib`:

```powershell
python scripts/entrenar_modelos.py
```
