
#!/bin/bash
# Enhanced Deployment script for DeliveryAI Prediction App

echo "🚀 Setting up DeliveryAI - AI-Powered Delivery Time Prediction Application..."
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Python is installed
print_step "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    print_error "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
print_status "Found Python $PYTHON_VERSION"

# Check if pip is installed
print_step "Checking pip installation..."
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    print_error "pip is not installed. Please install pip."
    exit 1
fi

# Create virtual environment
print_step "Creating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    $PYTHON_CMD -m venv venv
    source venv/Scripts/activate
    PIP_CMD="venv/Scripts/pip"
else
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
    PIP_CMD="venv/bin/pip"
fi

print_status "Virtual environment created and activated"

# Upgrade pip
print_step "Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install dependencies
print_step "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    $PIP_CMD install -r requirements.txt
    print_status "Dependencies installed successfully"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Check if model files exist
print_step "Checking model files..."
if [ ! -d "models" ]; then
    print_warning "Models directory not found. Creating directory..."
    mkdir -p models
    print_status "Please ensure model files are in the models/ directory before running the app"
fi

# Check if MLflow directory exists
print_step "Checking MLflow setup..."
if [ ! -d "mlruns" ]; then
    print_warning "MLflow tracking directory not found. Creating directory..."
    mkdir -p mlruns
    print_status "MLflow directory created"
fi

# Create necessary directories
print_step "Creating application directories..."
mkdir -p static/css static/js static/images
mkdir -p templates
mkdir -p logs
print_status "Application structure created"

# Set environment variables
print_step "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True
MLFLOW_TRACKING_URI=file:./mlruns
SECRET_KEY=your-secret-key-here
EOF
    print_status "Environment file created"
fi

# Check if Flask app exists
print_step "Checking Flask application..."
if [ ! -f "app.py" ]; then
    print_error "app.py not found. Please ensure the Flask application file exists."
    exit 1
fi

print_status "Flask application found"

# Run tests if test file exists
if [ -f "test_api.py" ]; then
    print_step "Running application tests..."
    $PYTHON_CMD test_api.py
fi

# Display startup information
echo ""
echo "=================================================================="
print_status "🎉 DeliveryAI Setup Complete!"
echo "=================================================================="
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "1. Start the application:"
echo -e "   ${BLUE}$PYTHON_CMD app.py${NC}"
echo ""
echo "2. Open your browser and navigate to:"
echo -e "   ${BLUE}http://localhost:5000${NC}"
echo ""
echo "3. For MLflow UI (in a separate terminal):"
echo -e "   ${BLUE}mlflow ui --backend-store-uri file:./mlruns${NC}"
echo -e "   ${BLUE}http://localhost:5000 (MLflow UI)${NC}"
echo ""
echo -e "${GREEN}Application Features:${NC}"
echo "• 🏠 Home - AI-powered delivery time prediction"
echo "• 📊 Dashboard - Real-time analytics and insights"
echo "• 🧪 MLflow - Model tracking and experimentation"
echo ""
echo -e "${YELLOW}Note:${NC} Make sure your model files are in the models/ directory"
echo -e "${YELLOW}Note:${NC} Run the Jupyter notebook first to train models if needed"
echo ""

# Option to start the application immediately
read -p "Would you like to start the application now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Starting DeliveryAI application..."
    echo "Application will be available at: http://localhost:5000"
    echo "Press Ctrl+C to stop the application"
    echo ""
    $PYTHON_CMD app.py
fi
