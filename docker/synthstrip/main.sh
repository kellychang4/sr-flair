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

has_subject=$( gsutil ls -d "gs://${BUCKET_NAME}/data/*" | grep "${SUBJECT_ID}" )
if [[ ${has_subject} ]]; then 
  # Download Data from GCS Bucket and Rename to Data
  echo "Processing: ${SUBJECT_ID}"
  if [[ -d "${HOME}/data" ]]; then rm -r "${HOME}/data"; fi; mkdir "${HOME}/data" 
  gsutil -q cp -r "gs://${BUCKET_NAME}/data/${SUBJECT_ID}/**" "${HOME}/data"

  # Perform Skull Stripping with mri_synthstrip
  echo "Skull Stripping..."
  input_list=$( find "${HOME}/data" -name "*.nii" -type f )
  for input in ${input_list[@]}; do
    output=$( echo "${input}" | sed -E "s/.nii/_stripped.nii.gz/" )
    mri_synthstrip -i "${input}" -o "${output}" 
  done

  # Upload Skull Stripped Images to GCS Bucket
  echo "Uploading synthstrip derivatives..."
  gsutil -q cp -r "${HOME}/data/*_stripped.nii.gz" \
    "gs://${BUCKET_NAME}/derivatives/${SUBJECT_ID}"
else
  echo "[ERROR] Requested subject '${SUBJECT_ID}' does not exists."
fi