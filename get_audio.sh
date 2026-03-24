#!/bin/sh 
#
user="$1"
URL=$(curl -sfS "http://localhost:8080/episodes/$user/latest" | sed -n 's/.*"audio_url":[[:space:]]*"\([^"]*\)".*/\1/p')
[ -n "$URL" ] && [ "$URL" != "null" ] || { echo "Episode not ready."; exit 1; }
curl -fSLo episode.mp3 "$URL"
#
