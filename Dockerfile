# Runs the experiment in a container so agent-written code can't touch the host.
#   docker build -t cheat-sandbox .
#   docker run --rm --env-file .env -v "$PWD/results_v2:/app/results_v2" cheat-sandbox --dry-run
# Arguments after the image name go to run_experiment.py.
FROM python:3.11-slim

WORKDIR /app
# Print progress, including the cost projection, as it happens instead of at exit.
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# Unprivileged user: agent code can write to its temp folder and the results folders only.
RUN useradd --create-home agent && mkdir -p results_v2 results_v2_mock \
    && chown agent results_v2 results_v2_mock
USER agent

ENTRYPOINT ["python", "run_experiment.py"]
