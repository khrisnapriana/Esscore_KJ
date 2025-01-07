from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2 as cv
import easyocr
import numpy as np
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import psutil  # Import psutil untuk memantau penggunaan RAM

# Konfigurasi logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fungsi untuk memantau penggunaan RAM
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # Konversi ke MB
    logger.info(f"Penggunaan RAM saat ini: {memory_usage:.2f} MB")

# Inisialisasi FastAPI
app = FastAPI()

# Fungsi untuk menggabungkan kata dalam satu baris (opsional)
def merge_words(results, threshold=30):
    merged_lines = []
    current_line = []
    last_y = None

    for result in results:
        bbox, text, _ = result
        if bbox is None or text is None:
            continue

        y = bbox[0][1]

        # Gabungkan kata-kata pada baris yang sama
        if last_y is None or abs(y - last_y) <= threshold:
            current_line.append(text)
        else:
            merged_lines.append(" ".join(current_line))
            current_line = [text]

        last_y = y

    if current_line:
        merged_lines.append(" ".join(current_line))

    return merged_lines

# Fungsi OCR untuk mendeteksi teks pada gambar
def ocr(image: np.ndarray) -> List[str]:
    # Inisialisasi EasyOCR dengan folder model yang sudah ada
    model_path = "model"  # Lokasi folder model di repository
    detect = easyocr.Reader(['id'], gpu=False, model_storage_directory=model_path)

    # Lakukan OCR
    results = detect.readtext(image, detail=1)

    if not results:
        return []

    # Urutkan berdasarkan koordinat y
    results = sorted(results, key=lambda x: x[0][0][1])

    # Gabungkan kata dalam satu baris
    merged_lines = merge_words(results)

    return merged_lines

# Tambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Izinkan semua domain (gunakan daftar domain untuk keamanan lebih baik)
    allow_credentials=True,  # Izinkan pengiriman kredensial (seperti cookie) jika diperlukan
    allow_methods=["*"],  # Izinkan semua metode HTTP (GET, POST, dll.)
    allow_headers=["*"],  # Izinkan semua header
)

# Endpoint untuk tes deploy
@app.get("/")
def test_deploy():
    return {"apimodelkunci deployed."}

# Endpoint untuk model kj
@app.post("/kj")
async def process_image(file: UploadFile = File(...)):
    log_memory_usage()  # Log penggunaan RAM

    logger.info(f"File diterima: {file.filename}")  # Menggunakan logger di sini

    # Validasi tipe file
    if not file.content_type.startswith("image/"):
        logger.error("File bukan gambar.")
        return JSONResponse(content={"error": "File yang diunggah bukan gambar."}, status_code=400)

    try:
        # Baca file gambar
        contents = await file.read()
        logger.info("Gambar berhasil dibaca.")
        
        # Decode gambar
        nparr = np.frombuffer(contents, np.uint8)
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        if image is None:
            logger.error("Gambar tidak valid atau gagal di-decode.")
            return JSONResponse(content={"error": "Gambar tidak valid."}, status_code=400)

        logger.info("Gambar berhasil di-decode, memulai OCR.")
        
        # Proses OCR
        detected_text = ocr(image)
        logger.info(f"Teks terdeteksi: {detected_text}")

        if not detected_text:
            logger.warning("Tidak ada teks yang terdeteksi.")
            return {"message": "Tidak ada teks yang terdeteksi."}
        
        # Kembalikan hasil deteksi teks
        return {"detected_text": detected_text}
    
    except Exception as e:
        logger.exception(f"Terjadi kesalahan: {e}")
        return JSONResponse(content={"error": "Kesalahan internal server."}, status_code=500)
