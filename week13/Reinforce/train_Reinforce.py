#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import environment as Env
import keras
import numpy as np
import random


# In[2]:


# 강화학습 인공신경망
class REINFORCE(tf.keras.Model):
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = keras.layers.Dense(24, activation = 'relu') # 은닉층 (unit 개수 : 30, 활성함수 : ReLU)
        self.fc2 = keras.layers.Dense(24, activation = 'relu') # 은닉층 (unit 개수 : 30, 활성함수 : ReLU)
        self.fc_out = keras.layers.Dense(action_size, activation = 'softmax') # 출력층 (action_size = 5, 상, 하, 좌, 우, 제자리)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy = self.fc_out(x)
        
        return policy


# In[3]:


# 그리드월드 예제에서의 딥살사 에이전트
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size # 상태의 크기 정의
        self.action_size = action_size # 행동의 크기 정의
        
        # REINFORCE 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        
        self.model = REINFORCE(self.action_size)
        self.optimizer = keras.optimizers.Adam(lr = self.learning_rate)
        self.states, self.actions, self.rewards = [], [], [] # 상태, 행동, 보상을 저장하기 위해 list 형식으로 정의
        
    # 정책을 통해 행동 선택
    def get_action(self, state):
        print("get_action 함수")
        print("state: ", state)
        print("model(state) : ", self.model(state))
        policy = self.model(state)[0] # 현재 상태를 입력해 정책 확률만 출력
        policy = np.array(policy) # list를 numpy.array로 변환
        print(policy)
        print("action : ", np.random.choice(self.action_size, 1, p = policy)[0], "\n\n")
        return np.random.choice(self.action_size, 1, p = policy)[0] # 확률을 적용한 random.choice 함수로 0~4 중에 한 수치를 선택
    
    def discount_rewards(self, rewards): # 반환 값 계산 함수(입력 : 저장한 reward, 출력 : 반환값 array)
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))): # 효율적으로 반환값을 계산하기 위해 거꾸로 진행
            running_add = running_add * self.discount_factor + rewards[t] # 반환값 = reward(t번째) + discount factor * 반환값
            discounted_rewards[t] = running_add # 반환값을 discounted_rewards라는 array에서 저장
        return discounted_rewards
    
    def append_sample(self, state, action, reward): # 한 에피소드 동안의 상태, 행동, 보상을 저장
        self.states.append(state[0]) # 상태 저장
        self.rewards.append(reward) # 보상 저장
        act = np.zeros(self.action_size) # 행동을 one hot encoding으로 변환
        act[action] = 1
        self.actions.append(act) # 행동을 저장
        
    def train_model(self): # 정책신경망 업데이트 함수        
        discounted_rewards = np.float32(self.discount_rewards(self.rewards)) # 보상을 discount_rewards 함수를 통해 반환값을 return하고 반환값을 numpy.float32형식으로 변환
        discounted_rewards -= np.mean(discounted_rewards) # 데이터를 Z-score 표준화 방법으로 정규화함(정책 신경망의 업데이트 성능 향상)
        discounted_rewards /= np.std(discounted_rewards)
        
        # 크로스 엔트로피 오류함수 계산
        model_params = self.model.trainable_variables
        
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            policies = self.model(np.array(self.states))
            actions = np.array(self.actions)
            action_prob = tf.reduce_sum(actions * policies, axis = 1)
            cross_entropy = -tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * discounted_rewards)
            entropy = -policies * tf.math.log(policies)
        
        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], [] # 상태, 행동, 보상 list 초기화
        return np.mean(entropy)


# In[4]:


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env.Env(render_speed = 0.01) # 환경 instance 생성 (게임 속도를 0.01로 조정)
    state_size = 15 # 상태 개수 정의
    action_space = [0, 1, 2, 3, 4] # 행동 정의
    action_size = len(action_space) # 행동 개수 정의
    agent = REINFORCEAgent(state_size, action_size) # REINFORCE instance 생성
    
    scores, episodes = [], []
    
    EPISODES = 100 # episode 횟수 정의.
    
    for e in range(EPISODES): 
        done = False
        score = 0
        step = 0
        
        # env 초기화
        state = env.reset() # 환경을 초기화하고 상태를 받음 (list 형식)
        state = np.reshape(state, [1, state_size]) # 상태 list를 (1, 15)의 numpy.array로 변환
        
        while not done: # episode가 끝나지 않으면 계속 실행
            # 몇 번째 스텝인지 확인
            step += 1
            print("step : {:d}".format(step))
            
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            
            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # 샘플로 모델 학습
            agent.append_sample(state, action, reward)
            score += reward
            state = next_state
            
            if done: # episode가 끝나면(goal에 도착하면)
#                print("step : {:d}".format(step))
                
                # 에피소드마다 정책신경망 업데이트
                entropy = agent.train_model()
                
                # 에피소드마다 학습결과 출력
                print("episode: {:3d} | score: {:3d} | entropy: {:.3f}\n".format(e, score, entropy))                
                #agent.__init__.self.model = tf.zeros(shape=None, name=None)
                
                
        # 10 에피소드마다 모델 저장
        if e % 10 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')

