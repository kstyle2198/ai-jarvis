#!/bin/bash

fastapi dev server.py &

streamlit run app_async1.py &

wait
