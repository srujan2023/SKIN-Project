$ErrorActionPreference = "Stop"

if (!(Test-Path ".\\venv\\Scripts\\python.exe")) {
  throw "Missing .\\venv. Create it first: python -m venv venv"
}

& .\venv\Scripts\python.exe -c "import tensorflow as tf; import tensorflow.python; print(tf.__version__)"
