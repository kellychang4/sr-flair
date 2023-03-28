# Activate Base Environment
FROM sr-flair/base

# Install Dependencies
RUN conda install -n base -c conda cmake
RUN pip install antspyx
RUN pip cache purge

# Copy Host Executables into Binaries Directory
WORKDIR "/usr/local/bin"
COPY "main.sh" "main.sh"
RUN chmod +x "main.sh"

# Start in Home Directory and Copy Host Files 
WORKDIR "${HOME}"
COPY "MNI152NLin2009cAsym" "MNI152NLin2009cAsym"
COPY "main.py" "main.py"

# Set Entrypoint
ENTRYPOINT [ "main.sh" ]