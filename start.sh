#!/bin/bash
echo "Starting app..."
echo "Python version:"
python --version
echo "Testing imports..."
python -c "from app.main import app; print('Import OK')"
echo "Launching uvicorn..."
uvicorn app.main:app --host 0.0.0.0 --port $PORT