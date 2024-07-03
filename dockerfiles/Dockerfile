# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.10

RUN apt-get update
RUN apt-get install -y libhdf5-dev 
RUN apt-get install -y libhdf5-serial-dev

# RUN pip install xgboost

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5001-6000

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=5060

# Run app.py when the container launches
# CMD flask run --host=0.0.0.0 --port=$FLASK_RUN_PORT
ENTRYPOINT flask run --host=0.0.0.0 --port=$FLASK_RUN_PORT


##Run instructions
##Build the image using the following command - 
# docker build -t flask-app .