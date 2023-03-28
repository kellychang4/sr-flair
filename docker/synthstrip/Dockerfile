# Activate SynthStrip Image
FROM freesurfer/synthstrip:1.4

# Activate Base Environment
FROM sr-flair/base

# Copy SynthSeg and Set Environmental Variables
COPY --from=0 "/freesurfer" "/opt/freesurfer"
ENV PATH="/opt/freesurfer:${PATH}"
ENV FREESURFER_HOME="/opt/freesurfer" 

# Install SynthStrip Dependencies
RUN pip install numpy surfa torch && \
    pip cache purge

# Copy Host Files into Binaries 
WORKDIR "/usr/local/bin"
COPY "main.sh" "main.sh"
RUN chmod +x "main.sh" 

# Start in Home Directory and Set Entrypoint
WORKDIR "${HOME}"
ENTRYPOINT [ "main.sh" ]