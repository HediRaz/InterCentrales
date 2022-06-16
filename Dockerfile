FROM nvcr.io/nvidia/pytorch:22.04-py3

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install tkinter for Python 3
RUN apt-get update && apt-get install -y python3-tk && rm -rf /var/lib/apt/lists/*
COPY . /workspace
