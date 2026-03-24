#!/bin/sh
#curl -sS -X POST "http://localhost:8000/users" \
#	  -H "accept: application/json" \
#	    -H "Content-Type: application/json" \
#	      --data '{ "schedule_time":"string", "topics":["Ottawa","Senators","Hockey"], "max_duration_min":7, "voice":"en_US" }'
USER_ID=$(
  curl -sS -X POST "http://localhost:8000/users" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    --data '{ "schedule_time":"string","topics":["Ottawa","Senators","Hockey"],"max_duration_min":3,"voice":"en_US" }' \
  | jq -r '.id'
)

echo "New user id: $USER_ID"

# or per-user (adjust if your API name differs)
#curl -sS -X POST http://localhost:8000/generate_episode \
#  -H "Content-Type: application/json" \
#  --data '{"user_id": $USER_ID}'
