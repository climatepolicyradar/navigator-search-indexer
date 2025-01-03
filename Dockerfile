FROM python:3.10
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN mkdir /app
WORKDIR /app

RUN apt-get update

# Copy the src module here so that poetry can install it as a package
COPY ./src ./src

# Install pip and poetry
RUN pip install --upgrade pip
RUN pip install poetry

# Create layer for dependencies
COPY ./poetry.lock ./pyproject.toml ./

# Install python dependencies using poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-root

# Copy files to image
COPY ./cli ./cli
COPY ./tests ./tests
COPY ./.git ./.git
COPY ./.pre-commit-config.yaml ./.flake8 ./.gitignore ./

# Pre-download the model
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Run the indexer on the input s3 directory
ENTRYPOINT [ "sh", "./cli/run.sh" ]
