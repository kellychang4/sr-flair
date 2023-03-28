# Activate Dockerfile Base
FROM --platform=linux/arm64 continuumio/miniconda3:22.11.1

# Set USER
USER "root"
ENV HOME="/home"

# Install System Packages
RUN apt-get update -q && \
    apt-get install -qy \
        build-essential \
        curl \
        libpng-dev && \
    apt-get clean

# Install Google Cloud SDK
WORKDIR /opt
ARG GOOGLE_SDK="google-cloud-cli-420.0.0-linux-arm.tar.gz"
RUN curl -sL "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${GOOGLE_SDK}" | \
    tar -xz && "google-cloud-sdk/install.sh" --quiet
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"
RUN gcloud components update --quiet

# Set "Home" and Maintain Bash Terminal Colors
RUN find "/root" -maxdepth 1 -not -wholename "/root" -execdir mv {} "${HOME}" \; && \
    sed -i "s/^# export LS_OPTIONS/export LS_OPTIONS/" "${HOME}/.bashrc" && \
    sed -i "s/^# eval/eval/" "${HOME}/.bashrc" && \
    sed -i "s/dircolors/dircolors --sh/" "${HOME}/.bashrc" && \
    sed -i "s/^# alias l/alias l/" "${HOME}/.bashrc"
WORKDIR "${HOME}"