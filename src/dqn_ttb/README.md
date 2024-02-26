# 1. Deep Q-Learning
코드는 로보티즈의 [머신러닝 튜토리얼](https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/#machine-learning) 코드와 [PyTorch(한국 사용자) DQN 튜토리얼](https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html)을 기반으로 만들어졌습니다.


### YouTube

[YouTube Link](https://youtu.be/DUBrjx43RE8?si=EaVsYQ6waJKiJW0l)

[YouTube Link](https://youtu.be/gVq_Z3rb1RY)


### Simulation
1. `ROS1` 패키지로 [Turtlebot3 Simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git)와 [현재 리포지토리](https://github.com/redEddie/turtlebot3_machine_learning.git)를 `git clone` 해주세요.
   ```
   cd ~/catkin_ws/src
   git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
   git clone https://github.com/redEddie/turtlebot3_machine_learning.git
   ```

1. (이미 적용되어있을 수 있음.)`/turtlebot3_machine_learning`패키지 밑의 `/nodes`의 파일들은 실행가능한 파일로 속성을 변환해주세요.

   ```
   chmod +x turtlebot3_dqn_stage_1
   ```

1. `burger`의 `.xacro`를 수정합니다. 이는 센서(LDS)의 정보를 수정하는 것으로 실제 로봇에 적용할 때는 그에 맞추어 다시 학습시켜야 합니다.

   DQN 코드는 [링크의 메뉴얼](https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/#set-state)을 따라합니다.
   
   ```
   roscd turtlebot3_description
   cd ./urdf
   gedit turtlebot3_burger.gazebo.xacro
   ```

1. `ROS` 패키지를 빌드해주세요.

   ```
   catkin build
   ```

1. 패키지를 소스해주세요.

   ```
   cd ~/catkin_ws
   source ./devel/setup.bash
   ```

1. Gazebo 환경에 `월드`와 `머신`을 로드해주세요.

   ```
   roslaunch turtlebot3_gazebo turtlebot3_dqn_stage_2.launch
   ```

1. 알고리즘을 동작시켜주세요.
   ```
   roslaunch dqn_ttb turtlebot3_dqn_stage_30.launch
   ```
