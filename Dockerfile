# Use a base Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask API code
COPY app.py .

# Copy the model files
COPY ml-model /app/ml-model

# Set environment variables if necessary
ENV MODEL_DIR=ml-model/
ENV MODEL_NAMES="ARM.h5 BRVO.h5 CRS.h5 CRVO.h5 CSR.h5 DN.h5 DR.h5 LS.h5 MH.h5 MYA.h5 ODC.h5 ODE.h5 ODP.h5 RPEC.h5 RS.h5 TSLN.h5"

# Expose the API port
EXPOSE 5000

# Start the Flask API
CMD ["python", "app.py"]
