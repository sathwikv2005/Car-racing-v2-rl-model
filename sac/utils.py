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

        curve_box_sh = h*0.2
        curve_box_eh = h*0.8
        curve_box_sw = w*0.2
        curve_box_ew = w*0.8

        roi = road_mask[int( curve_box_sh):int( curve_box_eh), int( curve_box_sw):int( curve_box_ew)]


        left = roi[:, :roi.shape[1]//2]
        right = roi[:, roi.shape[1]//2:]

        left_mean = np.mean(left)
        right_mean = np.mean(right)

        center_offset = abs(left_mean - right_mean)

        total = left_mean + right_mean + 1e-6  # avoid divide by zero
        curve_signal = (left_mean - right_mean) / total
        is_curve = abs(curve_signal)

        
        car_box_sh = h*0.67
        car_box_eh = h*0.8
        car_box_sw = w*0.47
        car_box_ew = w*0.53


        car_roi = grass_mask[int(car_box_sh):int(car_box_eh), int(car_box_sw):int(car_box_ew)]
        grass_ratio = np.mean(car_roi)
        on_road = grass_ratio < 0.3

        # Encourage staying on road 

        road_reward = (0.3 - grass_ratio)*2  
        shaped_reward += road_reward


        offroad_penalty = -3.0 * grass_ratio
        shaped_reward += offroad_penalty

        center_penality = (center_offset * 0.5) ** 1.5
        shaped_reward -= center_penality

        # Speed Reward (encourage forward motion)
       
        speed = float(state_info["speed"])

        # Smooth reward proportional to speed
        road_factor = max(0.0, 0.3 - grass_ratio)
        move_reward = (speed * road_factor / 20.0 ) ** 1.5

        high_speed_reward = 0
        if speed > 70 and on_road:
            high_speed_reward += (speed - 70) / 30.0

        shaped_reward += high_speed_reward
        shaped_reward += move_reward
        speed_reward = 0

        target_speed = 1.0 - curve_signal  # sharp turn = lower target speed
        # speed_error = (-abs(target_speed - (speed / 100.0)))/10

        # speed_reward = (1.0 - speed_error) * 0.5
        slow_penality = 0
        slow_threshold = 40
        # if speed < slow_threshold:
        #     slow_penality = (speed-slow_threshold) * 0.05

        shaped_reward += speed_reward 
        shaped_reward += slow_penality 

        gas_reward = gas * 0.1 * (1 - grass_ratio)
        shaped_reward += gas_reward


        offroad_speeding_penalty = 0
        if grass_ratio > 0.2:
            offroad_speeding_penalty = speed / 50.0
            shaped_reward -= offroad_speeding_penalty
       
        # Tile Progress Reward (primary objective)
       
        tile_progress = state_info["tile_progress"]

        if tile_progress > 0:
            tile_reward = tile_progress * 6.0 * (1 - grass_ratio)
            shaped_reward += tile_reward
            got_tile = 1
        else:
            tile_reward = 0.0
            got_tile = 0

       
        # Control Penalties (stability constraints)
       
        steer_penalty = abs(steer) * 0.05 # penalize jitter
        brake_penalty = brake * 0.005

        steer_change_penalty = abs(steer - prev_action[0]) * 0.1
        shaped_reward -= steer_change_penalty

        shaped_reward -= steer_penalty
        shaped_reward -= brake_penalty

       
        

       
        # Logging Information (structured for analysis)
       
        info = {
            # --- Reward breakdown ---
            "reward/base": base_reward,
            "reward/total": shaped_reward,
            "reward/road": road_reward,
            "reward/off-road": offroad_penalty,
            "reward/offroad_speeding_penalty": offroad_speeding_penalty,
            "reward/move": move_reward,
            "reward/high_speed_reward": high_speed_reward,
            "reward/speed_error": speed_reward,
            "reward/slow_penality": slow_penality,
            "reward/center_penality": center_penality,
            "reward/tile": tile_reward,
            "reward/steer_penalty": -steer_penalty,
            "reward/steer_change_penalty": -steer_change_penalty,
            "reward/brake_penalty": -brake_penalty,
            "reward/gas": gas_reward,

            # --- State metrics ---
            "state/speed": speed,
            "state/grass_ratio": grass_ratio,
            "state/tile_progress": tile_progress,
            "state/curve_signal": curve_signal,
            "state/target_speed": target_speed,

            # --- Behavior indicators ---
            "event/got_tile": got_tile,
            "event/off_road": int(not on_road),
            "event/stopped": int(speed < 0.5),

            # --- Action diagnostics ---
            "action/steer": float(steer),
            "action/gas": float(gas),
            "action/brake": float(brake),
        }

##################
        # Copy original frame
        debug = obs.copy()

        # ROI coordinates
        y1, y2 = int(car_box_sh), int(car_box_eh)
        x1, x2 = int(car_box_sw), int(car_box_ew)

        # Draw rectangle (BGR: Green)
        color = (0, 255, 0)
        if shaped_reward < 0:
            color = (0, 0, 255)

        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)  # cyan box

        # # --- Draw center split ---
        # mid_x = (x1 + x2) // 2
        # cv2.line(debug, (mid_x, y1), (mid_x, y2), (255, 0, 0), 2)  # blue line

        # text = f"{grass_ratio:.2f}"
        text = f"reward:{shaped_reward:.2f}"
        cv2.putText(
            debug,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA
        )

        scale = 4
        debug_large = cv2.resize(debug, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("ROI Debug", debug_large)
        cv2.waitKey(1)
#################

        return shaped_reward, info


