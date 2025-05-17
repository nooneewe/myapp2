import sys
import os
import yaml
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
import base64
import time
from io import BytesIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from onnxruntime.quantization import quantize_dynamic, QuantType

# ---------- 0. تسريع المعالجة متعدد الخيوط ----------
num_threads = os.cpu_count() or 1
os.environ['OMP_NUM_THREADS']      = str(num_threads)
os.environ['MKL_NUM_THREADS']      = str(num_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

# ---------- 1. المسار الأساسي والإعدادات ----------
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cfg = yaml.safe_load(open(os.path.join(BASE_DIR, 'configs.yaml'), 'r'))
MODEL_PATH = os.path.join(BASE_DIR, cfg['model_path'])
H          = cfg['height']    # 64
W_MIN      = cfg['width']     # 128
VOCAB      = list(cfg['vocab'])
BLANK_IDX  = len(VOCAB)

# ---------- 2. تكميم ONNX إلى INT8 إن لزم ----------
quantized_model = MODEL_PATH.replace('.onnx', '_int8.onnx')
if not os.path.exists(quantized_model):
    quantize_dynamic(
        MODEL_PATH,
        quantized_model,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['MatMul', 'Gemm']
    )

# ---------- 3. إعداد جلسة ONNX Runtime ثابتة ---------- 
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads      = num_threads
opts.inter_op_num_threads      = 1
opts.enable_mem_pattern        = True

session = ort.InferenceSession(
    quantized_model,
    sess_options=opts,
    providers=['CPUExecutionProvider']
)
inp_meta = session.get_inputs()[0]
out_name = session.get_outputs()[0].name

# ---------- 4. FastAPI Setup ---------- 
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    img: str  # Data URI base64

# ---------- 5. Helper Functions ----------

def load_three_frames_from_bytes(gif_bytes: bytes):
    gif = Image.open(BytesIO(gif_bytes))
    n = getattr(gif, 'n_frames', 1)
    idxs = [0, n//2, n-1] if n>1 else [0]
    frames = []
    for i in idxs:
        gif.seek(i)
        frames.append(np.array(gif.convert('RGB'), dtype=np.uint8))
    while len(frames) < 3:
        frames.append(frames[-1])
    return frames

def batch_preprocess(frames):
    proc = []
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    for f in frames:
        h,w = f.shape[:2]
        y0 = max((h - H)//2, 0)
        crop = f[y0:y0+H,:,:]
        if crop.shape[0] != H:
            pad_top = (H - crop.shape[0])//2
            pad_bot = H - crop.shape[0] - pad_top
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bot, 0, 0,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=(255,255,255))
        img_f = crop[..., ::-1].astype(np.float32)/255.0
        img_f = (img_f - mean[None,None,:]) / std[None,None,:]
        proc.append(img_f.transpose(2,0,1))
    return np.stack(proc, 0)

def ctc_decode_consensus(logits: np.ndarray) -> str:
    seqs = []
    for log in logits:
        ids, prev, out = log.argmax(1), BLANK_IDX, []
        for i in ids:
            if i!=BLANK_IDX and i!=prev: out.append(VOCAB[i])
            prev = i
        seqs.append(''.join(out))
    avg = logits.mean(0)
    ids, res, prev = avg.argmax(1), [], None
    for i in ids:
        if i==BLANK_IDX: continue
        c = VOCAB[i]
        if c==prev and all(s.count(c)==1 for s in seqs): continue
        res.append(c); prev=c
    return ''.join(res)

def predict_from_bytes(gif_bytes: bytes):
    frames = load_three_frames_from_bytes(gif_bytes)
    batch  = batch_preprocess(frames)
    # warm-up run before timing
    session.run([out_name], {inp_meta.name: batch})
    t0 = time.perf_counter()
    probs = session.run([out_name], {inp_meta.name: batch})[0]
    elapsed = (time.perf_counter() - t0) * 1000
    text = ctc_decode_consensus(probs)
    return text, elapsed

# ---------- 6. Warmup at Startup ----------

@app.on_event('startup')
def do_warmup():
    warm_path = os.path.join(BASE_DIR, 'warmup.gif')
    if os.path.exists(warm_path):
        try:
            with open(warm_path,'rb') as f: gif_bytes = f.read()
            txt, ms = predict_from_bytes(gif_bytes)
            print(f"[Warmup] Prediction={txt}  Latency={ms:.2f}ms")
        except Exception as e:
            print("Warmup failed:", e)
    else:
        print("Warmup skipped; warmup.gif not found")

# ---------- 7. API Endpoint ----------

@app.post('/predict')
async def predict_image(data: ImageRequest):
    try:
        gif_bytes = base64.b64decode(data.img.split('base64,')[-1])
        text, t_ms = predict_from_bytes(gif_bytes)
        return {'status':'1','result':text,'time_ms':t_ms}
    except Exception as e:
        return {'status':'0','error':str(e)}
