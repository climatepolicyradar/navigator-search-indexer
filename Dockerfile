FROM python:3.9
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN mkdir /app
WORKDIR /app

RUN apt-get update

# Install pip and poetry
RUN pip install --upgrade pip
RUN pip install "poetry==1.2.2"

# Create layer for dependencies
COPY ./poetry.lock ./pyproject.toml ./

# Install python dependencies using poetry
RUN poetry config virtualenvs.create false
RUN poetry install

# Copy files to image
COPY ./src ./src
COPY ./cli ./cli
COPY ./tests ./tests

# Pre-download the model
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Run the indexer on the input s3 directory
ENTRYPOINT [ "sh", "./cli/run.sh" ]
