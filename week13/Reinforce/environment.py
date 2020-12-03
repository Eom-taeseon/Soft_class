#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time # agent의 움직임을 초단위로 보여주기 위한 time 패키지 import
import numpy as np # matrix 연산을 지원하는 numpy 패키지 import
import tkinter as tk #GUI 화면 구성을 위한 tkinter 패키지 import
from PIL import ImageTk, Image # 이미지 처리용 패키지 PIL import

PhotoImage = ImageTk.PhotoImage # Photo Image instance 생성
UNIT = 50  #  픽셀 수
HEIGHT = 5  # 그리드 세로
WIDTH = 5  # 그리드 가로

np.random.seed(1) # random seed 설정


class Env(tk.Tk): # 환경을 정의한 ENV 클래스 선언
    def __init__(self, render_speed = 0.01): # 클래스의 생성자 정의(게임 속도 설정)
        super(Env, self).__init__() # 다중 상속을 사용하는 하위 클래스가 MRO(Method Resolution Order)에서 올바른 다음 상위 클래스 함수를 호출함
        self.render_speed = render_speed # 게임 속도 초기화
        self.action_space = ['u', 'd', 'l', 'r'] # action space 초기화
        self.action_size = len(self.action_space) # action_size 초기화
        self.title('REINFORCE') # GUI 타이틀 초기화
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images() # load_images 함수에서 이미지를 읽어옴
        self.canvas = self._build_canvas() # Canvas 생성
        self.counter = 0 # step counter 초기화
        self.rewards = [] # reward 초기화
        self.goal = [] # goal 초기화
        # 장애물 설정
        self.set_reward([0, 1], -1)
        self.set_reward([1, 2], -1)
        self.set_reward([2, 3], -1)
        # 목표지점 설정
        self.set_reward([4, 4], 1)

    def _build_canvas(self): # _build_canvas 함수를 정의함
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT) # canvas 객체 생성,배경색을 'white'로 설정, height, width는 각각 5 * 50 = 250
        # 그리드 생성
        for c in range(0, WIDTH * UNIT, UNIT):  # 그리드 생성을 위한 세로줄 그리기
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 그리드 생성을 위한 가로줄 그리기
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        self.rewards = [] # 보상 초기화
        self.goal = [] # goal 초기화
        # 캔버스에 이미지 추가
        x, y = UNIT/2, UNIT/2 
        self.rectangle = canvas.create_image(x, y, image=self.shapes[0]) # rectangle 그리기

        # canvas 객체의 pack함수를 사용해 canvas 완성
        canvas.pack()

        return canvas

    def load_images(self): # 이미지를 PhotoImage 객체로 로드하는 load images함수를 정의함
        rectangle = PhotoImage(
            Image.open("rectangle.png").resize((30, 30)))
        triangle = PhotoImage(
            Image.open("triangle.png").resize((30, 30)))
        circle = PhotoImage(
            Image.open("circle.png").resize((30, 30)))

        return rectangle, triangle, circle # 생성한 PhotoImage 객체 반환

    def reset_reward(self): # 한 episode를 진행한 후 위치를 reset하는 함수 정의

        for reward in self.rewards: # reward에 저장한 canvas들을 삭제
            self.canvas.delete(reward['figure'])

        self.rewards.clear() # reward 초기화(장애물 3개, 목표지점 1개)
        self.goal.clear() # goal 초기화(목표지점만)
        self.set_reward([0, 1], -1) # 장애물 위치 초기화
        self.set_reward([1, 2], -1)
        self.set_reward([2, 3], -1)

        # goal위치 초기화
        self.set_reward([4, 4], 1)

    def set_reward(self, state, reward): # 위치와 보상을 입력하고 rewards 변수에서 저장하는 함수
        state = [int(state[0]), int(state[1])] # 위치를 list 형식으로 저장
        x = int(state[0]) # canvas에 있는 위치
        y = int(state[1])
        temp = {} # dict 형식으로, 보상, canvas image 방향 등 변수를 임시저장
        if reward > 0: # reward가 0보다 크다면
            temp['reward'] = reward # temp중에 key가 reward라는 변수를 통해 reward를 저장함
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                       (UNIT * y) + UNIT / 2,
                                                       image=self.shapes[2]) # canvas image를 figure에서 저장

            self.goal.append(temp['figure']) # 도착한 canvas image를 goal에서 저장함


        elif reward < 0: # reward가 0보다 작은 경우 (장애물이 있음)
            temp['direction'] = -1 # 방향을 -1로 저장
            temp['reward'] = reward # reward 저장
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[1]) # image 저장

        temp['coords'] = self.canvas.coords(temp['figure']) # canvas image의 bounding box(x1, y1, x2, y2)를 저장
        temp['state'] = state # 상태 저장
        self.rewards.append(temp) # 모든 정보를 reward list에서 append

    # new methods

    def check_if_reward(self, state): # 상태를 입력으로 목표지점에 도착할 수 있는지 확인
        check_list = dict() # dict 정의
        check_list['if_goal'] = False # 판단 변수 초기화
        rewards = 0 # 보상 초기화

        for reward in self.rewards: # 3개 장애물, 1개 목표지점을 for로 iteration함
            if reward['state'] == state: # 지금의 위치와 장애물 혹은 목표지점에 도착한다면
                rewards += reward['reward'] # 보상받기
                if reward['reward'] == 1: # 목표지점에 도착한 경우 if_goal을 True로 변경
                    check_list['if_goal'] = True 

        check_list['rewards'] = rewards # 보상 저장

        return check_list

    def coords_to_state(self, coords): # canvas 위치를 그리드월드 위치로 변환하는 함수
        x = int((coords[0] - UNIT / 2) / UNIT) # x 위치 계산
        y = int((coords[1] - UNIT / 2) / UNIT) # y 위치 계산
        return [x, y]

    def reset(self): # 환경 reset 함수
        self.update() # class update
        time.sleep(0.5) # 0.5초 멈춘다.
        x, y = self.canvas.coords(self.rectangle) # 출발 지점의 위치를 계산
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y) # canvas에서 출발 지점 그리기
        self.reset_reward() # reset_reward 함수를 통해 장애물과 도착지점 초기화
        return self.get_state() # get_state 함수를 통해 모든 정보가 들어가 있는 상태를 계산

    def step(self, action): # 한 time step 진행한 함수
        self.counter += 1 # time step 횟수 +1
        self.render() # 게임 진행(0.01초 sleep)

        if self.counter % 2 == 1: # 2번마다 reward 계산
            self.rewards = self.move_rewards()

        next_coords = self.move(self.rectangle, action) # 행동을 입력해 rectangle의 위치를 계사ㅏㄴ
        check = self.check_if_reward(self.coords_to_state(next_coords)) # 목표지점에 도착하는지 판단
        done = check['if_goal'] # 목표 지점 도착 판단 받기
        reward = check['rewards'] # reward 도착 판단 받기

        self.canvas.tag_raise(self.rectangle) # 행동에 따라 이동된 rectangle을 canvas의 top level에 표시

        s_ = self.get_state()

        return s_, reward, done

    def get_state(self): # 정보를 합해서 상태로 저장

        location = self.coords_to_state(self.canvas.coords(self.rectangle)) # coords_to_state를 통해 현재의 그리드월드 위치를 계산
        agent_x = location[0] # 좌표 x 받음
        agent_y = location[1] # 좌표 y 받음

        states = list() # 상태 list 초기화

        for reward in self.rewards: # 장애물, 목표지점 정보를 for로 읽어옴
            reward_location = reward['state'] # 장애물(목표지점)의 그리드월드 위치
            states.append(reward_location[0] - agent_x) # 장애물(목표지점)과 현재 위치의 상대 위치 x
            states.append(reward_location[1] - agent_y) # 장애물(목표지점)과 현재 위치의 상대 위치 y
            if reward['reward'] < 0: # 보상이 0보다 작으면
                states.append(-1) # -1과
                states.append(reward['direction']) # 방향을 append
            else: # 보상이 0보다 크면
                states.append(1) # 목표지점에 도착했기 때문에 1만 (append)

        return states # 상태 list return

    def move_rewards(self): # step을 진행한 후 reward 다시 계산
        new_rewards = [] # reward list 선언
        for temp in self.rewards: # 기존의 reward를 읽어오기
            if temp['reward'] == 1: # 목표지점인 경우
                new_rewards.append(temp)
                continue
            temp['coords'] = self.move_const(temp) # move_const 함수를 통해 이동된 장애물의 canvas image 위치를 다시 계산
            temp['state'] = self.coords_to_state(temp['coords']) # canvas 좌표를 그리드월드 좌표로 변환
            new_rewards.append(temp) # 장애물 정보를 저장
        return new_rewards # new_reward를 반환함

    def move_const(self, target): # 장애물의 움직임을 판단하는 함수

        s = self.canvas.coords(target['figure']) # 장애물의 위치

        base_action = np.array([0, 0]) # action array 초기화

        if s[0] == (WIDTH - 1) * UNIT + UNIT / 2: # 장애물의 방향이 오른쪽을 향하는 경우
            target['direction'] = 1 # 오른쪽으로 이동
        elif s[0] == UNIT / 2: # 장애물의 방향이 왼쪽인 경우
            target['direction'] = -1 # 왼쪽으로 이동

        if target['direction'] == -1: # 왼쪽으로 이동
            base_action[0] += UNIT
        elif target['direction'] == 1: # 오른쪽으로 이동 
            base_action[0] -= UNIT

        if (target['figure'] is not self.rectangle # 장애물 위치와 rectangle 위치가 다르고
           and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]): # 목표 지점에 도착한 경우
            base_action = np.array([0, 0]) # action array 초기화

        self.canvas.move(target['figure'], base_action[0], base_action[1]) # 장애물의 canvas 위치를 변경

        s_ = self.canvas.coords(target['figure']) # 이동한 위치를 저장

        return s_

    def move(self, target, action): # rectangle로 행동에 따라 이동하고 위치를 반환하는 함수
        s = self.canvas.coords(target) # rectangle 위치 받기

        base_action = np.array([0, 0]) # 행동 초기화

        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(target, base_action[0], base_action[1]) # rectangle을 행동에 따라 canvas에서 이동

        s_ = self.canvas.coords(target) # 이동된 위치 저장

        return s_

    def render(self): # 게임 속도를 조정한 함수
        time.sleep(0.07) # render_speed 변수에 따라 멈추게 됨
        self.update() # update

