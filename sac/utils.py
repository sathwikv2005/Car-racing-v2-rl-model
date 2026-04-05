import numpy as np
import cv2

class RewardWrapper:
    """
    Custom reward shaping for CarRacing-v2.

    Design principles:
    - Preserve original environment reward (main learning signal)
    - Add dense shaping rewards (road alignment, speed)
    - Strongly reinforce tile progress
    - Apply light control penalties to stabilize behavior
    """

    def __init__(self):
        # Step counter for optional periodic diagnostics
        self.step_count = 0


    def reset(self):
        """Reset episode-specific state."""
        self.step_count = 0

    def custom_reward(self, obs, reward, action, prev_action, state_info):
        

        self.step_count += 1

        # Unpack action
        steer, gas, brake = action

        # Handle tuple observations (Gymnasium compatibility)
        if isinstance(obs, tuple):
            obs = obs[0]

       
        # Base Reward (preserve original environment signal)
       
        base_reward = float(reward)
        shaped_reward = base_reward

       
        # Road Detection (from pixel colors)
       
        # Separate RGB channels
        r = obs[:, :, 0]
        g = obs[:, :, 1]
        b = obs[:, :, 2]

        # Identify grass vs road
        grass_mask = (g > 120) & (g > r + 20) & (g > b + 20)
        road_mask = ~grass_mask

        # Focus on region ahead of the car
        h, w = road_mask.shape

        curve_box_sh = h*0.4
        curve_box_eh = h*0.8
        curve_box_sw = w*0.3
        curve_box_ew = w*0.7

        roi = road_mask[int( curve_box_sh):int( curve_box_eh), int( curve_box_sw):int( curve_box_ew)]


        left = roi[:, :roi.shape[1]//2]
        right = roi[:, roi.shape[1]//2:]

        left_mean = np.mean(left)
        right_mean = np.mean(right)

        total = left_mean + right_mean + 1e-6  # avoid divide by zero
        curve_signal = (left_mean - right_mean) / total
        is_curve = abs(curve_signal)

        
        car_box_sh = h*0.67
        car_box_eh = h*0.8
        car_box_sw = w*0.47
        car_box_ew = w*0.53


        car_roi = grass_mask[int(car_box_sh):int(car_box_eh), int(car_box_sw):int(car_box_ew)]
        grass_ratio = np.mean(car_roi)
# ##################
#         # Copy original frame
#         debug = obs.copy()

#         # ROI coordinates
#         y1, y2 = int(car_box_sh), int(car_box_eh)
#         x1, x2 = int(car_box_sw), int(car_box_ew)

#         # Draw rectangle (BGR: Green)
#         color = (0, 255, 0)
#         if grass_ratio > 0.3:
#             color = (0, 0, 255)
#         cv2.rectangle(debug, (x1, y1), (x2, y2), color, 1)

#         text = f"{grass_ratio:.2f}"
#         cv2.putText(
#             debug,
#             text,
#             (x1, y1 - 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.4,
#             color,
#             1,
#             cv2.LINE_AA
#         )

#         scale = 4  # try 3–6 depending on your screen
#         debug_large = cv2.resize(debug, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

#         cv2.imshow("ROI Debug", debug_large)
#         cv2.waitKey(1)
# #################
        # Encourage staying on road 

        road_reward = (0.3 - grass_ratio)*2  
        shaped_reward += road_reward


        offroad_reward = 0
        if grass_ratio > 0.3:
            offroad_reward = -1
        shaped_reward += offroad_reward

        # Speed Reward (encourage forward motion)
       
        speed = float(state_info["speed"])

        # Smooth reward proportional to speed
        move_reward = speed / 150.0  
        shaped_reward += move_reward


        speed_reward = 0

        # target_speed = 1.0 - curve_signal  # sharp turn = lower target speed
        # speed_error = (-abs(target_speed - (speed / 100.0)))/10

        # speed_reward = (1.0 - speed_error) * 0.5
        slow_penality = 0
        if speed < 20:
            slow_penality = (speed-20) * 0.05

        shaped_reward += speed_reward 
        shaped_reward += slow_penality 

        gas_reward = gas * 0.1
        shaped_reward += gas_reward

       
        # Tile Progress Reward (primary objective)
       
        tile_progress = state_info["tile_progress"]

        if tile_progress > 0:
            tile_reward = tile_progress * 10.0
            shaped_reward += tile_reward
            got_tile = 1
        else:
            tile_reward = 0.0
            got_tile = 0

       
        # Control Penalties (stability constraints)
       
        steer_penalty = abs(steer) * 0.03 # penalize jitter
        brake_penalty = brake * 0.005

        shaped_reward -= steer_penalty
        shaped_reward -= brake_penalty

       
        

       
        # Logging Information (structured for analysis)
       
        info = {
            # --- Reward breakdown ---
            "reward/base": base_reward,
            "reward/total": shaped_reward,
            "reward/road": road_reward,
            "reward/off-road": offroad_reward,
            "reward/move": move_reward,
            "reward/speed_error": speed_reward,
            "reward/slow_penality": slow_penality,
            "reward/tile": tile_reward,
            "reward/steer_penalty": -steer_penalty,
            "reward/brake_penalty": -brake_penalty,
            "reward/gas": gas_reward,

            # --- State metrics ---
            "state/speed": speed,
            "state/grass_ratio": grass_ratio,
            "state/tile_progress": tile_progress,
            "state/curve_signal": curve_signal,
            # "state/target_speed": target_speed,

            # --- Behavior indicators ---
            "event/got_tile": got_tile,
            "event/off_road": int(grass_ratio > 0.3),
            "event/stopped": int(speed < 0.5),

            # --- Action diagnostics ---
            "action/steer": float(steer),
            "action/gas": float(gas),
            "action/brake": float(brake),
        }

        return shaped_reward, info


