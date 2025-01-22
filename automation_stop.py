import requests

# Konfigurasi API
API_URL_START = "https://api.idcloudhost.com/v1/sgp01/user-resource/vm/start"
API_URL_STOP = "https://api.idcloudhost.com/v1/sgp01/user-resource/vm/stop"
API_KEY = "xTlCA6jqvp5fQiuXF2lfdBKhBB3ytXBY"
SERVER_ID = "a98cd9f6-8f78-4e7f-b277-ef9a76c319c6"

# Fungsi untuk mengontrol VM
def control_vm(action):
    # Pilih URL berdasarkan aksi
    url = API_URL_START if action == "start" else API_URL_STOP
    headers = {
        "apikey": API_KEY,  # Header API Key
    }
    payload = {
        "uuid": SERVER_ID  # Parameter VM ID
    }
    try:
        # Kirim permintaan POST ke API
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            print(f"VM berhasil {action}.")
        else:
            print(f"Error {action} VM: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# Contoh penggunaan
if __name__ == "__main__":
    control_vm("stop")
