# Use official Python 3.12.8 base image
FROM python:3.12.8

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV GROQ_API_KEY=${GROQ_API_KEY}

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]