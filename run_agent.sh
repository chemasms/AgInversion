#!/bin/bash

# Navegar al directorio del proyecto
cd /home/chema/ProyectosPython/agente_inversion

# Activar el entorno virtual
source .venv/bin/activate

# Ejecutar el script de Python
python scripts/agente.py

# Desactivar el entorno virtual (opcional)
deactivate
