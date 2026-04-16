import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import pygame
import os

# ------------------------------
# Val
# ------------------------------
width = 1200
height = 720
BAR_HEIGHT = height // 10

game_state = "START"

start_hold_time = 0
hold_duration = 3  # second

time_add = 5
time_per_pic = 10

game_start_time = 0
time_limit = 99999
time_left = time_limit
time_bonus = 0
final_time_used = 0

passed_levels = 0

status = ""
game_clear = False

model = YOLO("yolo26n-pose.pt")
pygame.mixer.init()

#sound
sfx_pass = pygame.mixer.Sound("sfx/correct.mp3")
sfx_win = pygame.mixer.Sound("sfx/victory.mp3")
sfx_lose = pygame.mixer.Sound("sfx/fail.mp3")
bgm_intro = "sfx/intro.mp3"
bgm_bgm = "sfx/bgm.mp3"

current_bgm = None

sfx_win.set_volume(0.3)
sfx_lose.set_volume(0.3)

played_win = False
played_lose = False

max_error = 2300

cooldown = 0

poses = []
target_scales = []
target_poses = []
pictures = []

t_pose_path = "pose/T-Pose/t-pose.jpg"
quiz_paths = []
for file in os.listdir("pose/quiz"):
    if file.endswith(".jpg"):
        quiz_paths.append(os.path.join("pose/quiz", file))

quiz_paths.sort()
# pose_paths = [f"pose/pose{i}.jpg" for i in range(1, 23)]

#img
pic_win = cv.imread("ui/win.png", cv.IMREAD_UNCHANGED)
pic_lose = cv.imread("ui/game_over.png", cv.IMREAD_UNCHANGED)
pic_cat_lose = cv.imread("ui/game_over_cat.png", cv.IMREAD_UNCHANGED)
pic_cat_win = cv.imread("ui/win_cat.png", cv.IMREAD_UNCHANGED)

pic_logo = cv.imread("ui/logo.png", cv.IMREAD_UNCHANGED)
pic_logo = cv.resize(pic_logo, (0,0), fx=0.4, fy=0.4)

# camera
cap = cv.VideoCapture(0)

# ------------------------------
# Calculate functions
# ------------------------------

def get_accuracy(error, max_error):
    acc = max(0, 100 - (error / max_error) * 100)
    return int(acc)

def normalize_pose(pose):
    center = pose[0]  # nose
    return pose - center

def pose_error(p1, p2):
    return np.linalg.norm(p1 - p2)

def get_body_scale(pose):
    left_shoulder = pose[5]
    right_shoulder = pose[6]
    return np.linalg.norm(left_shoulder - right_shoulder)

def align_pose(target, player, target_scale):
    player_center = player[0]
    player_scale = get_body_scale(player)

    if target_scale == 0:
        target_scale = 1

    scale_ratio = player_scale / target_scale

    return target * scale_ratio + player_center

# ------------------------------
# Game functions
# ------------------------------

def check_status(frame, error, current_level, cooldown, total_level):
    accuracy = get_accuracy(error, max_error)

    if accuracy >= 78 and cooldown == 0:
        status = "PASS"
        current_level += 1
        cooldown = 20
    else:
        status = "FAIL"

    cv.putText(frame, f"{status} | Accuracy: {int(accuracy)} | Level: {int(current_level)}/{int(total_level + 1)}", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if status=="PASS" else (0,0,255), 2)
    return current_level, cooldown

def display_pic(frame, picture):
    h, w, _ = picture.shape
    x_offset= 15
    frame[(height//2) - (h//2): (height//2) + (h//2) , x_offset:x_offset+w] = picture

def is_t_pose(player_pose, target_pose_norm):
    error = pose_error(player_pose, target_pose_norm)
    acc = get_accuracy(error, max_error)
    return acc > 83

def draw_text_center_x(img, text, y, scale=1, color=(255,255,255), thickness=2):
    font = cv.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - text_size[0]) // 2
    cv.putText(img, text, (x, y), font, scale, color, thickness)

def draw_keypoints(target_pose_norm, keypoints, target_scale):
    aligned_target = align_pose(target_pose_norm, keypoints, target_scale)

    for i, (x, y) in enumerate(keypoints):
        cv.circle(frame, (int(x), int(y)), 5, (0,255,0), -1)

    for (x, y) in aligned_target:
        cv.circle(frame, (int(x), int(y)), 8, (0,0,255), -1)

def restart_game():
    global game_state, current_level, passed_levels, start_hold_time, game_start_time, cooldown, played_win, played_lose, time_bonus

    game_state = "PLAYING"
    current_level = 0
    passed_levels = 0
    start_hold_time = 0
    game_start_time = time.time()
    cooldown = 0
    played_win = False
    played_lose = False
    time_bonus = 0

def handle_restart(player_pose, current_time):
    global start_hold_time

    if is_t_pose(player_pose, t_pose_norm):

        if start_hold_time == 0:
            start_hold_time = current_time

        hold_time = current_time - start_hold_time
        countdown = int(hold_duration - hold_time) + 1
        countdown = max(0, countdown)

        draw_text_center_x(frame, f"Restarting in: {countdown}", (height//2) + 120, 2, (0,0,0), 9)
        draw_text_center_x(frame, f"Restarting in: {countdown}", (height//2) + 120, 2, (0,255,255), 4)

        if hold_time >= hold_duration:
            restart_game()

    else:
        draw_text_center_x(frame, "Hold T-Pose for 3 seconds to Restart", (height//2) + 120, 1.5, (0,0,0), 7)
        draw_text_center_x(frame, "Hold T-Pose for 3 seconds to Restart", (height//2) + 120, 1.5, (255,255,0), 3)
        start_hold_time = 0

def overlay_png(background, overlay, x, y):
    h, w = overlay.shape[:2]

    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0

    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            background[y:y+h, x:x+w, c] * (1 - mask) +
            overlay_img[:, :, c] * mask
        )

    return background

def play_bgm(path, loop=True):
    global current_bgm
    if current_bgm != path:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play(-1 if loop else 0)
        current_bgm = path

def stop_bgm():
    global current_bgm
    pygame.mixer.music.stop()
    current_bgm = None

# ------------------------------
# Initiate
# ------------------------------

#T-Pose pic
t_pose_img = cv.imread(t_pose_path)
t_pose_img = cv.resize(t_pose_img, (805, 1000))

results_pic = model(t_pose_img)

t_pose_norm = None
t_pose_scale = None

if results_pic[0].keypoints is not None:
    keypoints = results_pic[0].keypoints.xy[0]
    t_pose_norm = normalize_pose(keypoints)
    t_pose_scale = get_body_scale(keypoints)

#Quiz pic
for path in quiz_paths:
    img = cv.imread(path)
    img = cv.resize(img, (805, 1000))

    results_pic = model(img)

    if results_pic[0].keypoints is not None:
        keypoints = results_pic[0].keypoints.xy[0]

        poses.append(img)
        target_poses.append(normalize_pose(keypoints))

        target_scales.append(get_body_scale(keypoints))

        pic_small = cv.resize(img, (0,0), fx=0.4, fy=0.4)
        pictures.append(pic_small)

current_level = 0
total_levels = len(poses)
# time_limit = time_per_pic*total_levels
time_limit = 5
time_left = time_limit

# ------------------------------
# Game Loop
# ------------------------------

while cap.isOpened():
    current_time = time.time()

    success, frame = cap.read()
    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (width,height))
    if success:
        # Run tracking with persist=True to maintain IDs across frames
        results = model.track(frame, persist=True)

        if game_state == "START":
            play_bgm(bgm_intro)
            overlay_png(frame, pic_logo, width//2 - pic_logo.shape[1] // 2, pic_logo.shape[0] // 8)
            draw_text_center_x(frame, "Hold T-Pose for 3 seconds to Start", height//2 + 100, 1.5, (0,0,0), 7)
            draw_text_center_x(frame, "Hold T-Pose for 3 seconds to Start", height//2 + 100, 1.5, (255,255,0), 3)

            if results[0].keypoints is not None and results[0].keypoints.xy is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0]
                player_pose = normalize_pose(keypoints)

                draw_keypoints(t_pose_norm, keypoints, t_pose_scale)

                if is_t_pose(player_pose, t_pose_norm):

                    if start_hold_time == 0:
                        start_hold_time = current_time

                    hold_time = current_time - start_hold_time
                    countdown = int(hold_duration - hold_time) + 1

                    countdown = max(0, countdown)

                    #show countdown
                    draw_text_center_x(frame, f"Starting in: {countdown}", height//2 + 200, 2, (0,0,0), 9)
                    draw_text_center_x(frame, f"Starting in: {countdown}", height//2 + 200, 2, (0,255,255), 4)

                    if hold_time >= hold_duration:
                        game_state = "PLAYING"
                        game_start_time = time.time()
                        current_level = 0
                        passed_levels = 0

                else:
                    start_hold_time = 0
                    draw_text_center_x(frame, "Adjust your T-Pose", height//2 + 200, 1.5, (0,0,0), 7)
                    draw_text_center_x(frame, "Adjust your T-Pose", height//2 + 200, 1.5, (0,0,255), 3)
            else:
                draw_text_center_x(frame, "NO PLAYER", height//2 + 200, 1.5, (0,0,0), 7)
                draw_text_center_x(frame, "NO PLAYER", height//2 + 200, 1.5, (0,0,255), 3)

        elif game_state == "PLAYING":
            play_bgm(bgm_bgm)
            cv.rectangle(frame, (0, 0), (width, BAR_HEIGHT), (0, 0, 0), -1)

            elapsed = time.time() - game_start_time
            time_left = time_limit + time_bonus - elapsed

            #timer text
            cv.putText(frame, f"Time: {int(time_left)}", (900,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            if time_left <= 0:
                game_state = "GAMEOVER"
                final_time_used = time.time() - game_start_time
                continue

            if cooldown > 0:
                cooldown -= 1

            #win the game
            if current_level >= total_levels:
                game_state = "RESULT"
                final_time_used = time.time() - game_start_time
                continue

            target_scale = target_scales[current_level]
            target_pose_norm = target_poses[current_level]
            picture = pictures[current_level]

            if results[0].keypoints is not None and results[0].keypoints.xy is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0]

                draw_keypoints(target_pose_norm, keypoints, target_scale)

                player_pose = normalize_pose(keypoints)
                error = pose_error(player_pose, target_pose_norm)

                prev_level = current_level
                current_level, cooldown = check_status(frame, error, current_level, cooldown, total_levels-1)

                #pass the level
                if current_level > prev_level:
                    passed_levels += 1
                    time_bonus += time_add  #add time when ever pass the level
                    sfx_pass.play()

            else:
                cv.putText(frame, "NO PLAYER", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            display_pic(frame, picture)

        elif game_state == "RESULT":

            #play sound
            stop_bgm()
            if not played_win:
                sfx_win.play()
                played_win = True

            overlay_png(frame, pic_win, width//2 - pic_win.shape[1] // 2, pic_win.shape[0] // 2)
            overlay_png(frame, pic_cat_win, 0, height - pic_cat_win.shape[0])

            draw_text_center_x(frame, f"Time Used: {int(final_time_used)} second!", height//2, 1, (0,0,0), 5)
            draw_text_center_x(frame, f"Time Used: {int(final_time_used)} second!", height//2, 1, (255,255,255), 2)

            draw_text_center_x(frame, f"{passed_levels}/{total_levels}", (height//2) + 50,  1, (0,0,0), 5)
            draw_text_center_x(frame, f"{passed_levels}/{total_levels}", (height//2) + 50,  1, (255,255,255), 2)

            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0]

                draw_keypoints(t_pose_norm, keypoints, t_pose_scale)
                player_pose = normalize_pose(keypoints)

                handle_restart(player_pose, current_time)
            

        elif game_state == "GAMEOVER":
            #play sound
            stop_bgm()
            if not played_lose:
                sfx_lose.play()
                played_lose = True

            overlay_png(frame, pic_lose, width//2 - pic_lose.shape[1] // 2, pic_lose.shape[0] // 2)

            draw_text_center_x(frame, "You ran out of time", height//2, 1, (0,0,0), 5)
            draw_text_center_x(frame, "You ran out of time", height//2, 1, (255,255,255), 2)

            draw_text_center_x(frame, f"{passed_levels}/{total_levels}", (height//2) + 50,  1, (0,0,0), 5)
            draw_text_center_x(frame, f"{passed_levels}/{total_levels}", (height//2) + 50,  1, (255,255,255), 2)

            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0]

                draw_keypoints(t_pose_norm, keypoints, t_pose_scale)
                player_pose = normalize_pose(keypoints)

                handle_restart(player_pose, current_time)

            overlay_png(frame, pic_cat_lose, width - pic_cat_lose.shape[1], height // 2 - pic_lose.shape[0])

        cv.imshow("game", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()

