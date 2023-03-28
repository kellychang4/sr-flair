#!/usr/bin/env bash

# Ensure Subject ID Provided
SUBJECT_ID="${1}"
if [[ -z ${SUBJECT_ID} ]]; then
  echo "[ERROR] Subject ID is required as first positional argument."
  exit 1
fi

# Configure Google Cloud Platform Services
gcloud auth activate-service-account --key-file "${CREDENTIAL_KEYFILE}"
gcloud config set project "${PROJECT_ID}"

# Check Subject for Derivatives 
has_derivatives=$( gsutil ls -d "gs://${BUCKET_NAME}/derivatives/*" | grep "${SUBJECT_ID}" )
if [[ ${has_derivatives} ]]; then
  # Begin Coregistration Progress
  echo "Processing: ${SUBJECT_ID}"

  # Download Data from GCS Bucket and Rename to Data
  echo "  Downloading Derivatives..."
  gsutil -q cp -r "gs://${BUCKET_NAME}/derivatives/${SUBJECT_ID}" "${HOME}"
  gsutil -q cp "gs://${BUCKET_NAME}/files/subject-sidedness_latest.csv" "${HOME}"
  if [[ -d "${HOME}/data" ]]; then rm -r "${HOME}/data"; fi
  mv "${HOME}/${SUBJECT_ID}" "${HOME}/data"

  # Perform Image Coregistration
  echo "  Coregistering Images..."
  python "${HOME}/main.py" "${SUBJECT_ID}"

  # Upload Coregistered Image to GCS Bucket
  echo "  Uploading Coregistered Images..."
  gsutil -q cp -n "${HOME}/data/**" "gs://${BUCKET_NAME}/derivatives/${SUBJECT_ID}"
else
  echo "[ERROR] Requested subject ${SUBJECT_ID} does not have derivatives."
fi