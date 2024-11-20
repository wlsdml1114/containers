import gradio as gr
import cv2
import numpy as np
import os
import math
import subprocess
from tqdm import tqdm
import pickle
import re
import torch
from scipy.ndimage import gaussian_filter1d
import shutil
import ffmpeg
from datetime import datetime

# var init
current_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def headpose_pred_to_degree(pred):
    """
    pred: (bs, 66) or (bs, 1) or others
    """
    if pred.ndim > 1 and pred.shape[1] == 66:
        # NOTE: note that the average is modified to 97.5
        device = pred.device
        idx_tensor = [idx for idx in range(0, 66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)
        pred = F.softmax(pred, dim=1)
        degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 97.5

        return degree

    return pred

def get_rotation_matrix(pitch_, yaw_, roll_):
    PI = np.pi
    """ the input is in degree
    """
    # transform to radian
    pitch = pitch_ / 180 * PI
    yaw = yaw_ / 180 * PI
    roll = roll_ / 180 * PI

    device = pitch.device

    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = torch.ones([bs, 1]).to(device)
    zeros = torch.zeros([bs, 1]).to(device)
    x, y, z = pitch, yaw, roll

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([bs, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([bs, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)  # transpose

def transform_keypoint(kp_info: dict, pitch, yaw, roll, scale):
    """
    transform the implicit keypoints with the pose, shift, and expression deformation
    kp: BxNx3
    """
    kp = torch.tensor(kp_info['kp']).to(device)    # (bs, k, 3)
    # pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

    t, exp = torch.tensor(kp_info['t']).to(device), torch.tensor(kp_info['exp']).to(device)
    # scale = kp_info['scale']

    pitch = headpose_pred_to_degree(pitch)
    yaw = headpose_pred_to_degree(yaw)
    roll = headpose_pred_to_degree(roll)

    bs = kp.shape[0]
    if kp.ndim == 2:
        num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
    else:
        num_kp = kp.shape[1]  # Bxnum_kpx3

    rot_mat = get_rotation_matrix(pitch, yaw, roll)    # (bs, 3, 3)

    # Eqn.2: s * (R * x_c,s + exp) + t
    kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
    kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

    return kp_transformed

def extract_frame(video_file, frame_number):
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        return None, "Error: Cannot open video file"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 요청된 프레임이 비디오 길이를 초과하면 마지막 프레임으로 설정
    if frame_number >= total_frames:
        frame_number = total_frames - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        return None, "Error: Cannot read the frame"

    cap.release()

    # BGR에서 RGB로 변환 (Gradio에서 이미지를 제대로 표시하기 위해 필요)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame_rgb, None

def pickle_refine(folder_path, save_folder_path):
    # 파일 리스트 불러오기
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    pkl_files_sorted = sorted(pkl_files, key=extract_number)

    # 모든 데이터를 하나로 결합
    all_data = {'motion': [], 'c_lip_lst': [], 'c_eyes_lst': [], 'n_frames': 0}
    file_data_lengths = []

    for file_name in pkl_files_sorted:
        with open(os.path.join(folder_path, file_name), 'rb') as file:
            data = pickle.load(file)
            
            # 데이터 전처리 및 결합
            data['motion'] = data['motion'][1:]
            data['c_lip_lst'] = data['c_lip_lst'][1:]
            data['c_eyes_lst'] = data['c_eyes_lst'][1:]
            data['n_frames'] = data['n_frames'] - 1

            all_data['motion'].extend(data['motion'])
            all_data['c_lip_lst'].extend(data['c_lip_lst'])
            all_data['c_eyes_lst'].extend(data['c_eyes_lst'])
            all_data['n_frames'] += data['n_frames']
            
            file_data_lengths.append(len(data['motion']))

    # 데이터 추출 및 smoothing
    scales = [motion['scale'][0].item() for motion in all_data['motion']]
    pitch = [motion['pitch'][0][0].cpu() for motion in all_data['motion']]
    yaw = [motion['yaw'][0][0].cpu() for motion in all_data['motion']]
    roll = [motion['roll'][0][0].cpu() for motion in all_data['motion']]
    exp_data = np.array([motion['exp'] for motion in all_data['motion']])
    exp_data = exp_data[:, 0, :, :]

    # 가우시안 스무딩 적용
    sigma = 99.0
    smoothed_scales = gaussian_filter1d(scales, sigma=sigma)
    smoothed_pitch = torch.tensor(gaussian_filter1d(np.array(pitch), sigma=sigma)).to(device)
    smoothed_yaw = torch.tensor(gaussian_filter1d(np.array(yaw), sigma=sigma)).to(device)
    smoothed_roll = torch.tensor(gaussian_filter1d(np.array(roll), sigma=sigma)).to(device)
    smoothed_exp5 = gaussian_filter1d(exp_data, sigma=5.0, axis=0)
    smoothed_exp10 = gaussian_filter1d(exp_data, sigma=10.0, axis=0)
    smoothed_exp50 = gaussian_filter1d(exp_data, sigma=50.0, axis=0)

    # 데이터를 파일별로 나누어 저장
    start_idx = 0

    for idx, file_name in enumerate(pkl_files_sorted):
        file_len = file_data_lengths[idx]
        file_data = {
            'motion': all_data['motion'][start_idx:start_idx + file_len],
            'c_lip_lst': all_data['c_lip_lst'][start_idx:start_idx + file_len],
            'c_eyes_lst': all_data['c_eyes_lst'][start_idx:start_idx + file_len],
            'n_frames': file_len
        }

        # 해당 파일의 데이터 업데이트
        for i in range(file_len):
            file_data['motion'][i]['scale'][0][0] = smoothed_scales[start_idx + i]
            R_i = get_rotation_matrix(smoothed_pitch[start_idx + i].unsqueeze(0), smoothed_yaw[start_idx + i].unsqueeze(0), smoothed_roll[start_idx + i].unsqueeze(0))
            file_data['motion'][i]['R'] = R_i.cpu().numpy().astype(np.float32)
            file_data['motion'][i]['x_s'] = transform_keypoint(file_data['motion'][i], smoothed_pitch[start_idx + i].unsqueeze(0), smoothed_yaw[start_idx + i].unsqueeze(0), smoothed_roll[start_idx + i].unsqueeze(0), torch.tensor(smoothed_scales[start_idx + i]).to(device))

            for j in range(21):
                file_data['motion'][i]['exp'][0,j,0] = smoothed_exp10[start_idx][j,0]
            file_data['motion'][i]['exp'][:, 3:5, 1] = smoothed_exp50[start_idx + i][3:5, 1]
            file_data['motion'][i]['exp'][:, 5, 2] = smoothed_exp50[start_idx + i][5, 2]
            file_data['motion'][i]['exp'][:, 8, 2] = smoothed_exp50[start_idx + i][8, 2]
            file_data['motion'][i]['exp'][:, 9, 1:] = smoothed_exp50[start_idx + i][9, 1:]
            file_data['motion'][i]['exp'][:, 3:5, 2] = smoothed_exp5[start_idx + i][3:5, 2]
            file_data['motion'][i]['exp'][:, 5, 1] = smoothed_exp5[start_idx + i][5, 1]
            file_data['motion'][i]['exp'][:, 8, 1] = smoothed_exp5[start_idx + i][8, 1]

        # 수정된 데이터를 다시 pkl 파일로 저장
        save_file_path = os.path.join(save_folder_path, file_name)
        with open(save_file_path, 'wb') as save_file:
            pickle.dump(file_data, save_file)
        print(f"Modified data has been saved to {save_file_path}")

        start_idx += file_len

def pickle_refine_each(folder_path, save_folder_path):
    # 파일 리스트 불러오기
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    pkl_files_sorted = sorted(pkl_files, key=extract_number)

    for file_name in pkl_files_sorted:
        with open(os.path.join(folder_path, file_name), 'rb') as file:
            data = pickle.load(file)
            

        # 데이터 추출 및 smoothing
        scales = [motion['scale'][0].item() for motion in data['motion']]
        pitch = [motion['pitch'][0][0].cpu() for motion in data['motion']]
        yaw = [motion['yaw'][0][0].cpu() for motion in data['motion']]
        roll = [motion['roll'][0][0].cpu() for motion in data['motion']]
        exp_data = np.array([motion['exp'] for motion in data['motion']])
        exp_data = exp_data[:, 0, :, :]

        # 가우시안 스무딩 적용
        sigma = 999.0
        smoothed_scales = gaussian_filter1d(scales, sigma=sigma)
        smoothed_pitch = torch.tensor(gaussian_filter1d(np.array(pitch), sigma=sigma)).to(device)
        smoothed_yaw = torch.tensor(gaussian_filter1d(np.array(yaw), sigma=sigma)).to(device)
        smoothed_roll = torch.tensor(gaussian_filter1d(np.array(roll), sigma=sigma)).to(device)
        smoothed_exp5 = gaussian_filter1d(exp_data, sigma=5.0, axis=0)
        smoothed_exp10 = gaussian_filter1d(exp_data, sigma=10.0, axis=0)
        smoothed_exp50 = gaussian_filter1d(exp_data, sigma=50.0, axis=0)

        for i in range(len(data['motion'])):
            data['motion'][i]['scale'][0][0] = smoothed_scales[i]
            # R_i = get_rotation_matrix(smoothed_pitch[i].unsqueeze(0), smoothed_yaw[i].unsqueeze(0), smoothed_roll[i].unsqueeze(0))
            # data['motion'][i]['R'] = R_i.cpu().numpy().astype(np.float32)
            # data['motion'][i]['x_s'] = transform_keypoint(data['motion'][i], smoothed_pitch[i].unsqueeze(0), smoothed_yaw[i].unsqueeze(0), smoothed_roll[i].unsqueeze(0), torch.tensor(smoothed_scales[i]).to(device))

            # for j in range(21):
            #     data['motion'][i]['exp'][0,j,0] = smoothed_exp10[0,j,0] # 0부터 테스트
            # data['motion'][i]['exp'][:, 3:5, 1] = smoothed_exp50[i][3:5, 1]
            # data['motion'][i]['exp'][:, 5, 2] = smoothed_exp50[i][5, 2]
            # data['motion'][i]['exp'][:, 8, 2] = smoothed_exp50[i][8, 2]
            # data['motion'][i]['exp'][:, 9, 1:] = smoothed_exp50[i][9, 1:]
            # data['motion'][i]['exp'][:, 3:5, 2] = smoothed_exp5[i][3:5, 2]
            # data['motion'][i]['exp'][:, 5, 1] = smoothed_exp5[i][5, 1]
            # data['motion'][i]['exp'][:, 8, 1] = smoothed_exp5[i][8, 1]

        # 수정된 데이터를 다시 pkl 파일로 저장
        save_file_path = os.path.join(save_folder_path, file_name)
        with open(save_file_path, 'wb') as save_file:
            pickle.dump(data, save_file)
        print(f"Modified data has been saved to {save_file_path}")


def get_video_files(directory, prefix, suffix):
    # 폴더 내의 모든 비디오 파일을 가져옵니다.
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(suffix) and not "concat" in f]
    sorted_files = sorted(files, key=lambda x: int(x.split(prefix)[1].split(suffix)[0]))
    return sorted_files

def concatenate_videos(folder_path, prefix, suffix, output_file):
        
    # 비디오 파일 목록 가져오기
    video_files = get_video_files(folder_path, prefix, suffix)

    # 첫 번째 비디오 파일을 사용하여 해상도와 FPS를 얻습니다.
    cap = cv2.VideoCapture(os.path.join(folder_path,video_files[0]))
    if not cap.isOpened():
        print(f"Error opening video file {video_files[0]}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # 비디오 작성기 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 *'XVID', *'DIVX'
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error initializing video writer")
        return

    # 모든 비디오 파일을 읽고 작성기에 작성합니다.
    for file in video_files:
        cap = cv2.VideoCapture(os.path.join(folder_path,file))
        if not cap.isOpened():
            print(f"Error opening video file {file}")
            continue

        # remove first frame
        ret, frame = cap.read()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"Output video file {output_file} created successfully.")

    
def fast_concatenate_videos(output_files, output_file):
        
    # # 비디오 파일 목록 가져오기
    # video_files = get_video_files(folder_path, prefix, suffix)

    # # 첫 번째 비디오 파일을 사용하여 해상도와 FPS를 얻습니다.
    cap = cv2.VideoCapture(output_files[0])
    if not cap.isOpened():
        print(f"Error opening video file {output_files[0]}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # 비디오 작성기 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 *'XVID', *'DIVX'
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error initializing video writer")
        return

    # 모든 비디오 파일을 읽고 작성기에 작성합니다.
    for file in output_files:
        cap = cv2.VideoCapture(file)
        if not cap.isOpened():
            print(f"Error opening video file {file}")
            continue

        # remove first frame
        ret, frame = cap.read()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"Output video file {output_file} created successfully.")


def process_video(video_file, workspace_dir, frame_number, image):
    # 워크스페이스 디렉토리 설정
    video_filename = os.path.basename(video_file)
    video_name, _ = os.path.splitext(video_filename)
    if workspace_dir == "":
        workspace_dir  = "./output"
    video_name = video_name +"_"+ datetime.now().strftime("%Y%m%d-%H%M%S")
    video_folder = os.path.join(workspace_dir, video_name)
    split_folder = os.path.join(video_folder, "1_split")
    wav_folder = os.path.join(video_folder, "2_wav")
    added_folder = os.path.join(video_folder, "3_added")
    motion_template_folder = os.path.join(video_folder, "4_motion_template")
    animated_folder = os.path.join(video_folder, "5_animated")
    result_folder = os.path.join(video_folder, "6_result")
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(split_folder, exist_ok=True)
    os.makedirs(wav_folder, exist_ok=True)
    os.makedirs(added_folder, exist_ok=True)
    os.makedirs(motion_template_folder, exist_ok=True)
    os.makedirs(animated_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)

    # WAV 생성
    wav_file = os.path.join(wav_folder, f"{video_name}.wav")

    # Convert MP4 to WAV
    if not os.path.exists(wav_file):
        # Convert MP4 to WAV
        ffmpeg.input(video_file).output(wav_file).run()
    else:
        print("WAV file already exists. Skipping conversion.")

    # OpenCV로 비디오 파일 로드
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        return None, "Error: Cannot open video file"

    # FPS와 프레임 크기 정보
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    # 비디오 길이 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps  # 초 단위
    num_segments = math.ceil(total_duration / 100)  # 전체 길이를 100초 단위로 나눠 필요한 segment 개수를 계산

    # 지정된 프레임 추출
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, first_frame = cap.read()
    if not ret:
        return None, f"Error: Cannot read frame {frame_number}"
    
    start_time = 0
    output_files = []

    for segment in tqdm(range(num_segments)):
        end_time = start_time + 100
        if end_time > total_duration:
            end_time = total_duration  # 마지막 segment는 총 길이를 넘지 않도록

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)
        output_file = os.path.join(split_folder, f"out_{segment:03d}.mp4")
        added_output_file = os.path.join(added_folder, f"{segment:03d}.mp4")

        # 비디오 라이터 설정
        # out_split = cv2.VideoWriter(output_file, codec, fps, (width, height))
        out_added = cv2.VideoWriter(added_output_file, codec, fps, (width, height))
        
        # 첫 프레임 추가
        first_frame_resized = cv2.resize(first_frame, (width, height))
        out_added.write(first_frame_resized)

        # 비디오 세그먼트 저장 및 추가 작업
        current_frame = start_time * fps
        while current_frame < end_time * fps:
            ret, frame = cap.read()
            if not ret:
                break
            # out_split.write(frame)
            out_added.write(frame)
            current_frame += 1
        
        # out_split.release()
        out_added.release()

        output_files.append(added_output_file)
        
        # 시작 시간을 다음 segment로 이동
        start_time += 100
    
    cap.release()

    # # 이미지 처리 및 motion template 생성
    # if image is not None:
    #     image_path = os.path.join(video_folder, "uploaded_image.png")
    #     image.save(image_path)

    #     # motion_template.py를 실행할 경로 설정
    #     python_path = f"{current_dir}/venv/Scripts/python.exe"  # Python 경로 (필요시 절대 경로로 수정)
    #     script_path = f"{current_dir}/motion_template.py"  # motion_template.py 스크립트 경로 (필요시 절대 경로로 수정)

    #     for name in tqdm(os.listdir(added_folder)):
    #         if not name.endswith("mp4"):
    #             continue

    #         video_path = os.path.join(added_folder, name)
    #         command = [
    #             python_path, script_path,
    #             '-d', video_path,
    #             '-s', image_path,
    #             '--scale', '2.5',
    #             '--source-max-dim', '1920'
    #         ]
            
    #         result = subprocess.run(command, capture_output=True, text=True)
    #         if result.returncode != 0:
    #             return f"Error processing {name}: {result.stderr}"
    
    # ################################### To do start ############################################
    # # motion smoothing
    # if smoothing_input :
    #     pickle_refine_each(added_folder, motion_template_folder)
    # else :
    #     for file_name in os.listdir(added_folder):
    #         if file_name.endswith('pkl'):
    #             shutil.copy(os.path.join(added_folder,file_name), os.path.join(motion_template_folder,file_name))
    # ################################### To do finish ###########################################

    # animating
    output_files = []
    if image is not None:
        image_path = os.path.join(video_folder, "uploaded_image.png")
        image.save(image_path)

        # motion_template.py를 실행할 경로 설정
        python_path = f"python"  # Python 경로 (필요시 절대 경로로 수정)
        script_path = f"{current_dir}/run.py"  # motion_template.py 스크립트 경로 (필요시 절대 경로로 수정)

        for name in tqdm(os.listdir(added_folder)):
            if not name.endswith("mp4"):
                continue

            video_path = os.path.join(added_folder, name)
            print(video_path)
            print(image_path)
            print(result_folder)
            command = [
                python_path, script_path,
                '--dri_video', video_path,
                '--src_image', image_path,
                '--out_folder', result_folder,
                '--paste_back'
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                return f"Error processing {name}: {result.stderr}"

            output_files.append(os.path.join(result_folder, f"{os.path.basename(image_path)}-{os.path.basename(video_path)}-org.mp4"))
    # output_file = os.path.join(result_folder, f"{os.path.basename(image_path)}-{os.path.basename(video_path)}-org.mp4")

    # 비디오 파일 병합
    # concatenate_videos(animated_folder, prefix, suffix, output_file)

    sorted_file_list = sorted(output_files, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    output_file = os.path.join(result_folder, f'{video_name}_merged_output.mp4')
    fast_concatenate_videos(sorted_file_list, output_file)

    # wav파일과 mp4 결합
    audio_output_file = os.path.join(result_folder, f'{video_name}_audio_merged_output.mp4')
    command = [
        'ffmpeg',
        '-y',
        '-i', output_file,
        '-i', wav_file,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        audio_output_file
    ]

    # subprocess를 사용하여 FFmpeg 명령 실행
    subprocess.run(command, check=True)
    return "Video processed and motion templates created successfully.", audio_output_file

# 그레이디오 인터페이스
def create_interface():
    with gr.Blocks() as demo:

        with gr.Tab("Video Processor"):
            gr.Markdown("### Video Processor with Motion Template Generator")

            with gr.Row():
                workspace_input = gr.Textbox(label="Workspace Directory", placeholder="Enter the path to your workspace directory")
            
            with gr.Row():
                image_input = gr.Image(label="Upload Image", type="pil")
                video_input = gr.Video(label="Upload Video")

            frame_input = gr.Number(label="Frame Number to Add", value=0, precision=0)
            
            process_button = gr.Button("Process Video and Create Motion Templates")
            output_video = gr.Video(label="Generated Video", interactive=False)
            result_text = gr.Textbox(label="Result")
            
            process_button.click(
                process_video,
                inputs=[video_input, workspace_input, frame_input, image_input],
                outputs=[result_text,output_video]
            )

        with gr.Tab("Uploaded Video Processor"):
            gr.Markdown("### Video Processor with Motion Template Generator")

            with gr.Row():
                workspace_input = gr.Textbox(label="Workspace Directory", placeholder="Enter the path to your workspace directory")
            
            with gr.Row():
                image_input = gr.Image(label="Upload Image", type="pil")
                video_input = gr.Textbox(label="Upload Video",  placeholder="Enter the path to your workspace directory")

            frame_input = gr.Number(label="Frame Number to Add", value=0, precision=0)
            
            process_button = gr.Button("Process Video and Create Motion Templates")
            output_video = gr.Video(label="Generated Video", interactive=False)
            result_text = gr.Textbox(label="Result")
            
            process_button.click(
                process_video,
                inputs=[video_input, workspace_input, frame_input, image_input],
                outputs=[result_text,output_video]
            )

        with gr.Tab("Frame Extractor"):
            gr.Markdown("### Upload a video and extract a specific frame")
            with gr.Row():
                video_input1 = gr.Video(label="Upload Video")
                output_image = gr.Image(label="Extracted Frame")
            frame_slider = gr.Slider(minimum=0, maximum=300, step=1, label="Frame Number (0-300)")
            # frame_number_input = gr.Number(value=0, label="Or Input Frame Number (0-300)")
            extract_button = gr.Button("Extract Frame")

            # 프레임 추출 함수 연결
            def update_frame(video, frame_number):
                if frame_number > 300:
                    frame_number = 300
                frame_rgb, error = extract_frame(video, int(frame_number))
                if error:
                    return None, error
                return frame_rgb
            
            extract_button.click(
                update_frame, 
                inputs=[video_input1, frame_slider], 
                outputs=[output_image]
            )
            
            # 슬라이더와 숫자 입력 동기화
            # frame_slider.change(lambda val: val, inputs=frame_slider, outputs=frame_number_input)
            # frame_number_input.change(lambda val: val, inputs=frame_number_input, outputs=frame_slider)

        

    return demo

# 인터페이스 실행
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(inbrowser=True,server_name="0.0.0.0")
