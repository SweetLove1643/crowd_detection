"""Upload ảnh thẳng MinIO S3 → trả snapshot_url (object_key thô) để đặt vào payload MQTT.

ENV:
  VMS_S3_ENDPOINT    http://192.168.1.250:18646
  VMS_S3_BUCKET      vms-captures
  VMS_S3_ACCESS_KEY  atin
  VMS_S3_SECRET_KEY  (cấp riêng)

Cài:  pip install boto3
Chạy: python s3_snapshot_url.py <category> <camera_id> <ảnh.jpg>
      → in ra snapshot_url, ví dụ: plate/2026-11-12/camera_<camera_id>/<uuid>.jpg
"""
import datetime
import os
import sys
import uuid

import boto3
from botocore.config import Config

VN_TZ = datetime.timezone(datetime.timedelta(hours=7))

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["VMS_S3_ENDPOINT"],
    aws_access_key_id=os.environ["VMS_S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["VMS_S3_SECRET_KEY"],
    region_name="us-east-1",
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                  connect_timeout=5, read_timeout=15,
                  retries={"max_attempts": 3, "mode": "standard"}),
)


def upload_snapshot(category: str, camera_id: str, image: bytes, ext: str = "jpg") -> str:
    """Trả snapshot_url = {category}/{YYYY-MM-DD giờ VN}/camera_{camera_id}/{uuid}.{ext}"""
    day = datetime.datetime.now(VN_TZ).strftime("%Y-%m-%d")
    snapshot_url = f"{category}/{day}/camera_{camera_id}/{uuid.uuid4().hex}.{ext}"
    s3.put_object(
        Bucket=os.environ.get("VMS_S3_BUCKET", "vms-captures"),
        Key=snapshot_url,
        Body=image,
        ContentType="image/png" if ext == "png" else "image/jpeg",
    )
    return snapshot_url


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    category, camera_id, path = sys.argv[1:]
    with open(path, "rb") as f:
        print(upload_snapshot(category, camera_id, f.read(), "png" if path.lower().endswith(".png") else "jpg"))
