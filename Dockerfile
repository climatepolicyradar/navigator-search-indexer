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

# Download the sentence transformer model
RUN mkdir /models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/msmarco-distilbert-dot-v5', cache_folder='/models')"

# Copy files to image
COPY ./data ./data
COPY ./src ./src
COPY ./cli ./cli

# Pre-download the model
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN python '/app/src/warm_up_model.py'

# Run the indexer on the input s3 directory
ENTRYPOINT [ "sh", "./cli/run.sh" ]