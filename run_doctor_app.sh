#!/bin/bash
# Launch MammoDoctor with the aicd conda environment
cd "$(dirname "$0")"
echo "Starting MammoDoctor on http://localhost:8501 ..."
/opt/anaconda3/envs/aicd/bin/streamlit run mammo_doctor.py --server.port 8501
