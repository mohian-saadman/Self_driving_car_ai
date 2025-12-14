# Self-Driving Car AI

This project is a simulation of a self-driving car that learns to navigate a race track using a deep neural network. The project uses Pygame for the simulation environment and PyTorch for the AI model.

## Features

*   **Realistic 2D Simulation:** A 2D racing environment with a car and a track.
*   **Deep Reinforcement Learning:** The car is trained using a Deep Q-Network (DQN) to make decisions based on sensor readings.
*   **Multiple Modes:**
    *   **Train:** Train a new AI model.
    *   **Test:** Test a trained model.
    *   **Manual:** Drive the car manually.
    *   **Race:** Race multiple AI-driven cars against each other.
*   **Interactive Menu:** A user-friendly menu to select different modes and options.
*   **Training Analysis:** Plots are generated to visualize the training progress.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mohian-saadman/Self_driving_car_ai.git
    cd Self_driving_car_ai
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the project, use the `main.py` script. The default mode is the menu.

```bash
python main.py
```

You can also specify a mode using the `--mode` argument:

```bash
python main.py --mode [train|test|manual|race]
```

### Menu

The menu allows you to:
*   **Train a new model:** You will be prompted to enter a name for your new model.
*   **Test a model:** Select a pre-trained model to see how it performs.
*   **Race models:** Select multiple models to race against each other.
*   **Drive manually:** Control the car with your keyboard.
*   **Quit:** Exit the application.

### Training

To train a new model, select the "Train" option from the menu or run:

```bash
python main.py --mode train --episodes <num_episodes>
```

The trained model will be saved in the `models/` directory, and training progress plots will be saved in the `plots/` directory.

### Testing

To test a model, select the "Test" option from the menu or run:

```bash
python main.py --mode test
```

This will load the `best_model.pth` by default.

## Project Structure

```
.
├── ai/
│   ├── model.py        # Neural network architecture
│   └── trainer.py      # Training and testing logic
├── game/
│   ├── car.py          # Car physics and sensors
│   ├── environment.py  # Pygame simulation environment
│   ├── menu.py         # Interactive menu
│   ├── track.py        # Race track
│   └── utils.py        # Utility functions
├── models/
│   └── best_model.pth  # Example of a trained model
├── plots/
│   └── ...             # Training progress plots
├── main.py             # Main entry point
└── requirements.txt    # Project dependencies
```

## Technologies Used

*   [Python](https://www.python.org/)
*   [Pygame](https://www.pygame.org/) for the simulation.
*   [PyTorch](https://pytorch.org/) for the neural network.
*   [NumPy](https://numpy.org/) for numerical operations.
*   [Matplotlib](https://matplotlib.org/) for plotting training progress.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
