''' PodWorker | modules | upload.py '''

import os
import time
from io import BytesIO

from PIL import Image
from boto3 import session
from botocore.config import Config


# --------------------------- S3 Bucket Connection --------------------------- #
bucket_session = session.Session()

boto_config = Config(
    signature_version='s3v4',
    retries={
        'max_attempts': 3,
        'mode': 'standard'
    }
)

if os.environ.get('BUCKET_ENDPOINT_URL', None) is not None:
    boto_client = bucket_session.client(
        's3',
        endpoint_url=os.environ.get('BUCKET_ENDPOINT_URL', None),
        aws_access_key_id=os.environ.get('BUCKET_ACCESS_KEY_ID', None),
        aws_secret_access_key=os.environ.get('BUCKET_SECRET_ACCESS_KEY', None),
        config=boto_config
    )
else:
    boto_client = None  # pylint: disable=invalid-name


# ---------------------------------------------------------------------------- #
#                                 Upload Image                                 #
# ---------------------------------------------------------------------------- #
def upload_image(job_id, job_result, result_index=0):
    '''
    Upload image to bucket storage.
    '''
    if boto_client is None:
        # Save the output to a file
        output = BytesIO()
        img = Image.open(job_result)
        img.save(output, format=img.format)

        os.makedirs("uploaded", exist_ok=True)
        with open(f"uploaded/{result_index}.png", "wb") as file_output:
            file_output.write(output.getvalue())

        return f"uploaded/{result_index}.png"

    output = BytesIO()
    img = Image.open(job_result)
    img.save(output, format=img.format)

    bucket = time.strftime('%m-%y')

    # Upload to S3
    boto_client.put_object(
        Bucket=f'{bucket}',
        Key=f'{job_id}/{result_index}.png',
        Body=output.getvalue(),
        ContentType="image/png"
    )

    output.close()

    presigned_url = boto_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': f'{bucket}',
            'Key': f'{job_id}/{result_index}.png'
        }, ExpiresIn=604800)

    return presigned_url
