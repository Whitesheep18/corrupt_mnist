# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy files to container
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY corrupt_mnist/ corrupt_mnist/
COPY data/ data/
COPY reports/ reports/
#COPY models/ models/

# set working directory 
#(As an alternative you can use RUN make requirements after copying Makefile)

WORKDIR /
RUN pip install -e . --no-cache-dir

# name our training script as the entrypoint
ENTRYPOINT ["python", "-u", "corrupt_mnist/train_model.py", "train"]

