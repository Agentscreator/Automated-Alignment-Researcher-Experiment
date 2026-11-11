# Runs the experiment in a container so agent-written code can't touch the host.
#   docker build -t cheat-sandbox .
#   docker run --rm --env-file .env -v "$PWD/results:/app/results" cheat-sandbox --runs 2
# Arguments after the image name go to run_experiment.py.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# Unprivileged user: agent code can write to its temp folder and /app/results only.
RUN useradd --create-home agent && mkdir -p results && chown agent results
USER agent

ENTRYPOINT ["python", "run_experiment.py"]
