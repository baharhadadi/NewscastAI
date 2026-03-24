#!/bin/sh 
#
sudo docker compose down; 
sudo docker comopose build;
sudo docker compose build --no-cache mcp; 
sudo docker compose up -d; 
sudo docker compose logs; 

