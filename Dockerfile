# Use a base Python image
FROM python:3.9
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python -m pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the ML models
COPY . /app

# Copy the Flask API code
COPY  . /app

# Set environment variables if necessary
ENV MODEL_NAMES="ARM.h5 BRVO.h5 CRS.h5 CRVO.h5 CSR.h5 DN.h5 DR.h5 LS.h5 MH.h5 MYA.h5 ODC.h5 ODE.h5 ODP.h5 RPEC.h5 RS.h5 TSLN.h5"


# Expose the API port
EXPOSE 5000
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Start the Flask API
CMD ["flask", "run"]


