## Rent Connectivity Scraper

This repository contains a Python-based web application designed to scrape rental listings and analyze their connectivity or location-based data. The project is structured as a Flask web application that automates the collection of rental information and presents it through a web interface.

---

## Project Structure

The repository is organized into a main application and a dedicated scraper module:

*   **`main.py`**: The entry point for the application.
*   **`rental_scraper/`**: The core package containing the application logic.
    *   **`app.py`**: The Flask application setup and route definitions.
    *   **`scraper.py`**: Contains the logic for web scraping rental data.
    *   **`connectivity.py`**: Handles connectivity analysis or geolocation-related tasks.
    *   **`templates/index.html`**: The front-end user interface for interacting with the scraper.
*   **`requirements.txt`**: Lists the Python dependencies required to run the project.
*   **`Procfile`**: Configuration for deploying the application (e.g., to platforms like Heroku).

---

## Features

*   **Web Scraping**: Automatically extracts rental listing data from targeted websites.
*   **Connectivity Analysis**: Evaluates listings based on specific connectivity or location metrics.
*   **Web Interface**: Provides a simple HTML dashboard to view or trigger scraping tasks.
*   **Deployment Ready**: Includes a `Procfile` for quick deployment to cloud hosting providers.

---

## Installation & Setup

### Prerequisites
*   Python 3
*   `pip` (Python package manager)

### Installation Steps
1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Rent_Connectivity
    ```

2.  **Install dependencies**:
    There are requirements files located in both the root and the `rental_scraper` directory. It is recommended to install from the root:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    Launch the app using the main entry point:
    ```bash
    python main.py
    ```

---

## Usage

Once the application is running, navigate to `http://localhost:8080` (or the port specified in your environment) in your web browser. Use the interface to input parameters for rental searches and view the processed results.
