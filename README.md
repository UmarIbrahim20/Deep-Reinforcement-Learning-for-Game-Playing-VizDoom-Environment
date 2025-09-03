# Deep Reinforcement Learning for Game Playing – VizDoom Environment

**Environment used:**  
The experiments were conducted on **VizDoomTakeCover-v0**, a first-person shooter scenario where agents must learn to survive and score points using only visual input.

---

## 📖 Project Overview
This project investigates **Deep Reinforcement Learning (DRL) agents** trained to play the VizDoom environment using raw image input.  
The goal was to compare different DRL algorithms (DQN, A2C, PPO) in terms of **average reward, game score, training time, and testing time**.  

Additionally, **Optuna** was employed for hyperparameter optimization, enabling automated fine-tuning of agents for improved performance.

---

## ⚙️ Methodology
### Preprocessing
- Input images resized and converted to grayscale to reduce dimensionality.  
- Normalization applied to focus on essential features and reduce computational load.  

### Algorithms Compared
1. **DQN (Deep Q-Network)**  
   - Value-based method estimating Q-values for state-action pairs.  
   - Suitable for discrete action spaces.  

2. **A2C (Advantage Actor-Critic)**  
   - Combines policy-based and value-based methods.  
   - Actor selects actions, critic evaluates them.  

3. **PPO (Proximal Policy Optimization)**  
   - Policy gradient method with stability in training.  
   - Handles large state-action spaces efficiently.  

### Hyperparameter Optimization
- **Optuna** used for automated hyperparameter search.  
- Improved efficiency and agent performance compared to manual tuning.  

---

## 📊 Results
### Before Hyperparameter Tuning
| Algorithm | Training Time (s) | Testing Time (s) | Mean Reward |
|-----------|-------------------|------------------|-------------|
| **PPO**  | 47.06 | 7.96 | 220.45 |
| **DQN**  | 15.45 | 7.33 | 181.41 |
| **A2C**  | 17.62 | 6.82 | 169.38 |

- PPO achieved the highest mean reward, but DQN was faster to train.  
- A2C showed lower performance compared to PPO and DQN.  

### After Hyperparameter Tuning (Optuna)
| Algorithm | Training Time (s) | Testing Time (s) | Mean Reward | Std Reward |
|-----------|-------------------|------------------|-------------|------------|
| **PPO**  | 37.82 | 10.18 | 264.05 | 127.33 |
| **DQN**  | 141.14 | 6.05 | 188.25 | 55.20 |
| **A2C**  | 168.26 | 7.23 | 191.95 | 57.01 |

- PPO showed the most significant improvement in performance.  
- DQN and A2C improved modestly after tuning but at higher computational cost.  

---

## 📌 Key Insights
- **PPO** is the most effective DRL agent for the VizDoom task, achieving the highest rewards after optimization.  
- **DQN** is faster initially but less effective in complex scenarios.  
- **A2C** provides a balance but did not outperform PPO.  
- **Optuna hyperparameter tuning** significantly enhances agent performance, especially for PPO.  

---

## 📌 References
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature. [DQN]  
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. [PPO]  
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press. [A2C foundation]  
