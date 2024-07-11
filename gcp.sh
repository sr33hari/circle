#!/bin/bash

eval "$(jq -r '@sh "PROJECT_ID=\(.project_id) REGION=\(.region) REPOSITORY_ID=\(.repository_id)"')"

repository=$(gcloud artifacts repositories describe $REPOSITORY_ID --location=$REGION --project=$PROJECT_ID --format=json)

if [[ $? -eq 0 ]]; then
  jq -n --arg name "$REPOSITORY_ID" '{name:"'$REPOSITORY_ID'"}'
else
  jq -n --arg name "" '{name:""}'
fi
