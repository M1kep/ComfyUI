FROM 12.2.0-runtime-ubuntu20.04

# Install python dependencies
WORKDIR /app
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 xformers
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

COPY . ./

ENTRYPOINT ["python", "/app/main.py", "--listen"]
