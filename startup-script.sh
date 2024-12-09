#!/bin/bash

# Update the package list and install Docker
apt-get update
apt-get install -y docker.io
systemctl enable --now docker

# Install Git (optional, if your app is stored in a Git repository)
apt-get install -y git

# Clone the Streamlit app repository (change the URL to your repository)
git clone https://github.com/yourusername/your-streamlit-app.git /home/streamlit-app

# Change to the project directory
cd /home/streamlit-app

# Build the Docker image
docker build -t streamlit-app .

# Run the Streamlit app
docker run -d -p 8501:8501 streamlit-app

echo "Streamlit app is running at http://$(hostname -I | awk '{print $1}'):8501"
