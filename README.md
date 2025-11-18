DTE-2502 — Neural Networks :: Graded Assignment 02

This repository contains my full implementation of the Deep Q-Learning agent for Snake, rewritten entirely in PyTorch as required by Graded Assignment 02 in DTE-2502 Neural Networks.

The original project provided by the teacher used TensorFlow/Keras.
This submission replaces the learning system with a PyTorch DQN, while preserving the environment, replay buffer structure, and visualization workflow.

RUNNING GRADED ASSIGNMENT 02
1. Clone the repository
   
   git clone https://github.com/codecrasher123/snake-rl-pytorch.git
   
   cd snake-rl-pytorch
   
2.Create and activate the Conda environment

   conda create -n uitnn python=3.11 -y
   
   conda activate uitnn
   
3. Install required libraries
   
   pip install -r requirements.txt
   
   To enable mp4 export
   
   pip install imageio-ffmpeg
   
4.Visualizing the trained agent(If you'd like to train the agent please run python training.py, if not you can visualise it)

  python game_visualization_torch.py
  
This will produce images/game_visual_v17.1_200000.mp4
5. Presentation video is in https://drive.google.com/file/d/122x9JPlmhxFLtDJqheqFvIVwUWjv4sfF/view?usp=sharing
6. Powerpoint presentation is in this main branch
