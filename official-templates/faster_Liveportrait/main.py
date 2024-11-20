from flask import Flask, request, jsonify
import os
import logging
from threading import Thread
from typing import Dict
import cv2
import math
import subprocess
from tqdm import tqdm
import re
from datetime import datetime
import asyncio
from typing import List, Optional

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

job_status: Dict[str, dict] = {}

# var init
current_dir = os.path.dirname(os.path.abspath(__file__))

def run_subprocess(command):
    """subprocess.run을 실행하는 별도의 함수"""
    return subprocess.run(command, capture_output=True, text=True)

async def process_video_one(python_path: str, script_path: str, video_path: str, 
                       image_path: str, result_folder: str) -> Optional[str]:
    """비동기로 단일 비디오를 처리하는 함수"""
    command = [
        python_path, script_path,
        '--dri_video', video_path,
        '--src_image', image_path,
        '--out_folder', result_folder,
        '--paste_back'
    ]
    
    # ThreadPoolExecutor를 사용하여 외부 프로세스 실행
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, run_subprocess, command)
        if result.returncode != 0:
            return f"Error processing {os.path.basename(video_path)}: {result.stderr}"
        
        return os.path.join(result_folder, 
                          f"{os.path.basename(image_path)}-{os.path.basename(video_path)}-org.mp4")
    except Exception as e:
        return f"Exception processing {os.path.basename(video_path)}: {str(e)}"


async def process_videos(image_path: str, added_folder: str, result_folder: str, 
                        current_dir: str, max_concurrent: int = 4) -> List[str]:
    """여러 비디오를 동시에 처리하는 메인 함수"""
    python_path = "python"
    script_path = f"{current_dir}/run.py"
    
    # 처리할 비디오 파일 목록 생성
    video_files = [f for f in os.listdir(added_folder) if f.endswith("mp4")]
    
    # 진행 상황을 표시할 tqdm 설정
    pbar = tqdm(total=len(video_files), desc="Processing videos")
    
    output_files = []
    tasks = []
    
    # 세마포어를 사용하여 동시 실행 수 제한
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(video_name):
        async with semaphore:
            video_path = os.path.join(added_folder, video_name)
            result = await process_video_one(python_path, script_path, video_path, 
                                      image_path, result_folder)
            pbar.update(1)
            if result and not result.startswith("Error") and not result.startswith("Exception"):
                output_files.append(result)
            return result
    
    # 모든 비디오에 대한 태스크 생성
    for video_name in video_files:
        task = asyncio.create_task(process_with_semaphore(video_name))
        tasks.append(task)
    
    # 모든 태스크 실행 및 결과 대기
    results = await asyncio.gather(*tasks)
    
    # 에러 확인 및 출력
    errors = [r for r in results if r and (r.startswith("Error") or r.startswith("Exception"))]
    if errors:
        print("\nErrors occurred during processing:")
        for error in errors:
            print(error)
    
    pbar.close()
    return output_files

async def lp_task(image_path, video_folder, added_folder, result_folder):
    # if image is not None:
    #     image_path = os.path.join(video_folder, "uploaded_image.png")
    #     image.save(image_path)
        
    output_files = await process_videos(
        image_path=image_path,
        added_folder=added_folder,
        result_folder=result_folder,
        current_dir=current_dir,
        max_concurrent=4  # 동시 처리할 최대 비디오 수
    )
    return output_files
    
def fast_concatenate_videos(output_files, output_file):
        
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


def process_video(video_file, image_path, workspace_dir="/workspace/Faster_liveportrait", frame_number=0):
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

    # animating
    output_files = asyncio.run(lp_task(image_path, video_folder, added_folder, result_folder))
    # if image_path is not None:
        # image_path = os.path.join(video_folder, "uploaded_image.png")
        # image.save(image_path)

        # motion_template.py를 실행할 경로 설정
    # python_path = f"python"  # Python 경로 (필요시 절대 경로로 수정)
    # script_path = f"{current_dir}/run.py"  # motion_template.py 스크립트 경로 (필요시 절대 경로로 수정)

    # for name in tqdm(os.listdir(added_folder)):
    #     if not name.endswith("mp4"):
    #         continue

    #     video_path = os.path.join(added_folder, name)
    #     print(video_path)
    #     print(image_path)
    #     print(result_folder)
    #     command = [
    #         python_path, script_path,
    #         '--dri_video', video_path,
    #         '--src_image', image_path,
    #         '--out_folder', result_folder,
    #         '--paste_back'
    #     ]
        
    #     result = subprocess.run(command, capture_output=True, text=True)
    #     if result.returncode != 0:
    #         return f"Error processing {name}: {result.stderr}"

    #     output_files.append(os.path.join(result_folder, f"{os.path.basename(image_path)}-{os.path.basename(video_path)}-org.mp4"))

    sorted_file_list = sorted(output_files, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    output_file = os.path.join(result_folder, f'{video_name}_merged_output.mp4')
    fast_concatenate_videos(sorted_file_list, output_file)

    return output_file

class VideoProcessor(Thread):
    def __init__(self, job_id: str, input_video: str, input_image: str):
        super().__init__()
        self.job_id = job_id
        self.input_video = input_video
        self.input_image = input_image
        
    def Faster_lp(self):
        """LivePortrait 처리 로직"""
        try:

            output_filename = process_video(self.input_video, self.input_image)
            return output_filename
            
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return False
        
    def run(self):
        job_status[self.job_id]['status'] = 'PROCESSING'
        success = self.Faster_lp()
        
        if success:
            job_status[self.job_id].update({
                'status': 'COMPLETED',
                'output_path': success
            })
        else:
            job_status[self.job_id].update({
                'status': 'FAILED',
                'error': 'Video processing failed'
            })

@app.route('/process', methods=['POST'])
def Faster_Liveportrait():
    try:
        data = request.get_json()
        input_video = data['input_video']
        input_image = data['input_image']
            
        job_id = str(hash(input_video + str(os.path.getmtime(input_video))))
        
        job_status[job_id] = {
            'status': 'PENDING',
            'input_video': input_video,
            'input_image': input_image
        }
        
        processor = VideoProcessor(job_id, input_video, input_image)
        processor.start()
        
        return jsonify({
            "job_id": job_id,
            "status": "PENDING"
        })
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in job_status:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_status[job_id])

@app.route('/rename', methods=['POST'])
def rename_file():
    try:
        data = request.get_json()
        if not data or 'old_path' not in data or 'new_path' not in data:
            return jsonify({"error": "Both old_path and new_path are required"}), 400
            
        old_path = data['old_path']
        new_path = data['new_path']
        
        if not os.path.exists(old_path):
            return jsonify({"error": "File not found"}), 404
            
        os.rename(old_path, new_path)
        return jsonify({
            "status": "success",
            "old_path": old_path,
            "new_path": new_path
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    os.makedirs('/workspace/output', exist_ok=True)
    app.run(host='0.0.0.0', port=8000)