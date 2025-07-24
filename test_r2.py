import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os

# Tải biến môi trường từ file .env
load_dotenv()

# Lấy các thông tin cấu hình
endpoint_url = os.getenv("R2_ENDPOINT_URL")
access_key_id = os.getenv("R2_ACCESS_KEY_ID")
secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
bucket_name = os.getenv("R2_BUCKET_NAME")

# In ra để kiểm tra xem đã đọc đúng chưa
print("--- Đang kiểm tra cấu hình ---")
print(f"Endpoint: {endpoint_url}")
print(f"Bucket: {bucket_name}")
print(f"Access Key ID: {'Có' if access_key_id else 'Thiếu!'}")
print("----------------------------\n")

if not all([endpoint_url, access_key_id, secret_access_key, bucket_name]):
    print("Lỗi: Một hoặc nhiều biến môi trường R2 bị thiếu trong file .env. Vui lòng kiểm tra lại.")
else:
    try:
        s3_client = boto3.client(
            service_name='s3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name='auto'
        )

        print(f"Đang thử kết nối và liệt kê các file trong bucket '{bucket_name}'...")
        # Lệnh list_objects_v2 là một cách tốt để kiểm tra kết nối và quyền
        response = s3_client.list_objects_v2(Bucket=bucket_name)

        if 'Contents' in response:
            print("\n>>> KẾT NỐI THÀNH CÔNG! <<<")
            print("Các file có trong bucket:")
            for obj in response['Contents']:
                print(f"- {obj['Key']} (Size: {obj['Size']} bytes)")
        else:
            print("\n>>> KẾT NỐI THÀNH CÔNG nhưng bucket rỗng hoặc không có file nào.")

    except ClientError as e:
        print("\n>>> KẾT NỐI THẤT BẠI! <<<")
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == 'InvalidAccessKeyId':
            print("Lỗi: Access Key ID không hợp lệ.")
        elif error_code == 'SignatureDoesNotMatch':
            print("Lỗi: Secret Access Key không chính xác.")
        elif error_code == 'NoSuchBucket':
            print(f"Lỗi: Bucket với tên '{bucket_name}' không tồn tại hoặc Endpoint URL sai.")
        else:
            print(f"Lỗi chi tiết từ client: {e}")
    except Exception as e:
        print(f"\n>>> Đã có lỗi không xác định xảy ra: {e}")