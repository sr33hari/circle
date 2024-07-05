# Use the official Python image from the Docker Hub
# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.10

RUN apt-get update
RUN apt-get install -y libhdf5-dev 
RUN apt-get install -y libhdf5-serial-dev

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV MONGO_URI=mongodb+srv://highhari:harilovesmongo@ciclecluster.1t5nbzl.mongodb.net/?retryWrites=true&w=majority&tls=true&appName=CicleCluster

# Run app.py when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "0", "main:app"]
