# src/utils/storage_client.py
import boto3
from botocore.exceptions import ClientError
from src.config import settings  # Sử dụng file settings tập trung
import os


# Cấu hình R2 - Đảm bảo các biến này có trong file .env và settings.py của bạn
# R2_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
# R2_ACCESS_KEY_ID=<READ_ONLY_ACCESS_KEY>
# R2_SECRET_ACCESS_KEY=<READ_ONLY_SECRET_KEY>
# R2_BUCKET_NAME=your-bucket-name

def get_r2_client():
    """Khởi tạo và trả về một client S3 đã được cấu hình cho R2."""
    try:
        return boto3.client(
            service_name='s3',
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            region_name='auto',
        )
    except Exception as e:
        print(f"Lỗi khi khởi tạo R2 client: {e}")
        return None


def download_file_from_r2(file_key: str, download_path: str) -> bool:
    """
    Tải một file từ R2 về một đường dẫn local.

    :param file_key: Tên file duy nhất trên R2 bucket.
    :param download_path: Đường dẫn local để lưu file.
    :return: True nếu tải thành công, False nếu thất bại.
    """
    s3_client = get_r2_client()
    if not s3_client:
        return False

    try:
        print(f"Đang tải file '{file_key}' từ bucket '{settings.R2_BUCKET_NAME}'...")
        s3_client.download_file(settings.R2_BUCKET_NAME, file_key, download_path)
        print(f"Tải file thành công, đã lưu tại: {download_path}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"Lỗi: File '{file_key}' không tồn tại trên bucket.")
        else:
            print(f"Lỗi từ Cloudflare R2: {e}")
        return False
    except Exception as e:
        print(f"Lỗi không xác định khi tải file: {e}")
        return False