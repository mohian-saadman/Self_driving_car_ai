# 🚗 Self-Driving Car AI

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-00B140?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

A 2D racing simulation where a car **teaches itself to drive** using Deep Reinforcement Learning — built entirely from scratch with PyTorch and Pygame.

> 💡 **No human-written driving rules. The car learns everything from scratch through trial, error, and reward signals.**

<!-- Add a GIF here once you record one: -->
<!-- ![Demo GIF](assets/demo.gif) -->

---

## 🧠 How It Works

The car is controlled by a **Deep Q-Network (DQN)** — a reinforcement learning algorithm where the agent learns by interacting with the environment and maximising cumulative reward.

**Input:** 5 distance sensors (rays) detecting track boundaries  
**Output:** 4 actions — accelerate, brake, turn left, turn right  
**Reward:** +1 for each checkpoint passed, −100 for crashing  

Key RL techniques implemented:
- **Experience Replay** — stores past transitions in a memory buffer to break correlation between samples
- **Target Network** — a separate frozen network updated periodically to stabilise training
- **Epsilon-Greedy Exploration** — balances exploring new actions vs. exploiting learned knowledge
- **Batch Training** — samples random minibatches from memory each step

---

## ✨ Features

- **Realistic 2D Simulation** — car physics with momentum, turning radius, and wall collision
- **Deep Q-Network (DQN)** — learns from sensor readings, no hand-coded driving logic
- **4 Modes:**
  - `train` — train a new AI model from scratch
  - `test` — watch a trained model drive
  - `manual` — drive the car yourself with keyboard
  - `race` — pit multiple trained AI models against each other
- **Interactive Menu** — user-friendly mode selector on launch
- **Training Plots** — real-time graphs of rewards and loss saved to `/plots`

---

## 📁 Project Structure

```
.
├── ai/
│   ├── model.py        # DQN neural network architecture
│   └── trainer.py      # Training loop, replay buffer, target network
├── game/
│   ├── car.py          # Car physics and sensor rays
│   ├── environment.py  # Pygame simulation environment
│   ├── menu.py         # Interactive menu
│   ├── track.py        # Race track
│   └── utils.py        # Utility functions
├── models/
│   └── best_model.pth  # Pre-trained model checkpoint
├── plots/              # Training progress graphs
├── main.py             # Entry point
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/mohian-saadman/Self_driving_car_ai.git
cd Self_driving_car_ai

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Launch the interactive menu (default)
python main.py

# Or go straight to a mode
python main.py --mode train --episodes 500
python main.py --mode test
python main.py --mode manual
python main.py --mode race
```

Trained models are saved to `models/` and training plots to `plots/`.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| PyTorch | DQN neural network & training |
| Pygame | 2D simulation environment |
| NumPy | Sensor data & numerical ops |
| Matplotlib | Training progress plots |

---

## 📈 Training Results

> Add a screenshot of your training plot here once you have one.  
> Example: reward increasing over episodes = the car is learning.

<!-- ![Training Plot](plots/training_progress.png) -->

---

## 🤝 Connect

Built by **Md. Mohian Hasan Saadman**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/mohianhasan)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/mohian-saadman)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
