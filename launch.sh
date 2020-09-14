#!/bin/bash

echo "Launching and mounting local_databse"
docker run --rm   --name pg-docker -e POSTGRES_PASSWORD=docker -d -p 5432:5432 -v /data/jocabrer/Repos/multfs/docker/postgres:/var/lib/postgresql/data  postgres
