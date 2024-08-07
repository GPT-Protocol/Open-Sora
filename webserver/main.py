from fastapi import FastAPI, HTTPException
from minio import Minio
from pydantic import BaseModel
import os
import re
from uuid import uuid4
from dotenv import load_dotenv
from random import randint

load_dotenv()

app = FastAPI()
minio_client = Minio(endpoint=os.environ["MINIO_ENDPOINT"],
                     access_key=os.environ["MINIO_ACCESS_KEY"],
                     secret_key=os.environ["MINIO_SECRET_KEY"])

VIDEO_BUCKET = "test1"

class VideoGenerationReqBody(BaseModel):
    prompt: str

@app.post("/generate/video")
def generate_video(req_body: VideoGenerationReqBody):
    try:
        raw_prompt = req_body.prompt
        
        remove_sp_chars_regex = r'[^a-zA-Z0-9\s]'
    
        # Remove special characters
        cleansed_prompt = re.sub(remove_sp_chars_regex, '', raw_prompt)
        
        file_id = str(uuid4())
        
        generate_video_command = f'CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
        --num-frames 4s --resolution 240p --aspect-ratio 9:16 --fps 15 --seed {randint(0, 100)} --save-dir ./samples/file_{file_id}\
        --prompt "{cleansed_prompt}"'
        
        os.system(generate_video_command)
        filename = f"{file_id}.mp4"
        file_path = f"./samples/file_{file_id}/sample_0000.mp4"

        with open(file_path, 'rb') as file_data:
            minio_client.fput_object(
                bucket_name=VIDEO_BUCKET,
                object_name=filename,
                file_path=file_path
            )
            
        os.system(f'rm -rf ./samples/file_{file_id}')
        
        video_url = f'https://minio.gphelpers.com/{VIDEO_BUCKET}/{filename}'        
        return {"data": {"url": video_url}}

    except Exception as err:
        print("Error:", err)
        raise HTTPException(status_code=500, detail="Server error")
    
