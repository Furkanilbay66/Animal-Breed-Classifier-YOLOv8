"""
Animal Detection & Breed Identification Web App
- YOLOv8-cls  : 7 hayvan sınıfı (butterfly/cat/chicken/cow/dog/horse/sheep)
- EfficientNet-B4 (timm / ImageNet-1k) : ırk / tür tespiti
- FastAPI + modern web arayüzü
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ultralytics import YOLO
from PIL import Image
import io, os, urllib.request, json
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_model_data_config, create_transform
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Global model state
# ──────────────────────────────────────────────────────────────────────────────
yolo_model     = None
breed_model    = None
breed_transform= None
imagenet_labels= None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ANIMAL_EMOJI = {
    'butterfly': '🦋', 'cat': '🐈', 'chicken': '🐔',
    'cow': '🐄',       'dog': '🐕', 'horse':  '🐎', 'sheep': '🐑',
}

ANIMAL_TR = {
    'butterfly': 'Kelebek', 'cat': 'Kedi',  'chicken': 'Tavuk',
    'cow': 'İnek',          'dog': 'Köpek', 'horse':   'At', 'sheep': 'Koyun',
}

# ImageNet-1k: sınıf adlarında aranacak anahtar kelimeler (irk tespiti için)
BREED_KEYWORDS = {
    'dog': [
        'terrier','retriever','shepherd','hound','spaniel','pointer','setter',
        'collie','poodle','husky','labrador','bulldog','boxer','pinscher',
        'dachshund','dalmatian','chihuahua','corgi','samoyed','chow','malamute',
        'greyhound','mastiff','rottweiler','doberman','beagle','bloodhound',
        'basset','cocker','weimaraner','vizsla','akita','shiba','pug','pekingese',
        'maltese','bichon','briard','newfoundland','bernese','kuvasz','komondor',
        'great dane','irish wolfhound','saluki','cairn','lhasa','tibetan',
        'shih-tzu','papillon','leonberg','groenendael','malinois','bouvier',
        'schipperke','affenpinscher','norfolk','sealyham',
    ],
    'cat': ['tabby','persian cat','siamese','egyptian cat','tiger cat'],
    'chicken': ['hen','cock','partridge','quail','prairie chicken','bantam'],
    'cow': ['ox','zebu','bison','water buffalo','longhorn','highland cattle',
            'guernsey','holstein','hereford'],
    'horse': ['horse','sorrel','zebra','arabian','thoroughbred','appaloosa',
              'stallion','gelding'],
    'sheep': ['sheep','ram','ewe','angora','merino','suffolk','dorset'],
    'butterfly': ['butterfly','admiral','monarch','sulphur','lycaenid',
                  'ringlet','morpho','lacewing','copper','fritillary'],
}

IMAGENET_LABELS_PATH = Path("imagenet_classes.txt")
IMAGENET_LABELS_URL  = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)

# ──────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, breed_model, breed_transform, imagenet_labels

    print(f"\n{'='*55}")
    print(f"  Device : {DEVICE.upper()}")
    if DEVICE == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # 1) YOLOv8 classification model
    if os.path.exists('best.pt'):
        yolo_model = YOLO('best.pt')
        print("  YOLOv8 : best.pt yüklendi")
    else:
        print("  YOLOv8 : best.pt bulunamadı — önce train_classifier.py çalıştırın")

    # 2) ImageNet labels
    if not IMAGENET_LABELS_PATH.exists():
        print("  Labels : imagenet_classes.txt indiriliyor...")
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, IMAGENET_LABELS_PATH)
    with open(IMAGENET_LABELS_PATH, 'r', encoding='utf-8') as f:
        imagenet_labels = [l.strip() for l in f]
    print(f"  Labels : {len(imagenet_labels)} ImageNet sınıfı yüklendi")

    # 3) Breed model (EfficientNet-B4, ImageNet-1k)
    print("  Breed  : EfficientNet-B4 yükleniyor...")
    _m = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1000)
    _m.eval()
    if DEVICE == 'cuda':
        _m = _m.cuda()
    data_cfg        = resolve_model_data_config(_m)
    breed_transform = create_transform(**data_cfg, is_training=False)
    breed_model     = _m
    print("  Breed  : EfficientNet-B4 hazır")
    print(f"{'='*55}\n")

    yield  # app running

app = FastAPI(title="Animal Detector", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Hayvan isimleri
ANIMALS = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']

# ──────────────────────────────────────────────────────────────────────────────
# Breed detection helper
# ──────────────────────────────────────────────────────────────────────────────
def _breed_indices_for(animal: str) -> list[int]:
    if imagenet_labels is None:
        return []
    keywords = BREED_KEYWORDS.get(animal, [])
    indices = []
    for i, lbl in enumerate(imagenet_labels):
        lbl_lower = lbl.lower()
        if any(kw in lbl_lower for kw in keywords):
            indices.append(i)
    return indices


def detect_breed(image: Image.Image, animal: str) -> dict | None:
    if breed_model is None or imagenet_labels is None:
        return None
    try:
        tensor = breed_transform(image).unsqueeze(0)
        if DEVICE == 'cuda':
            tensor = tensor.cuda()
        with torch.no_grad():
            probs = F.softmax(breed_model(tensor), dim=1)[0]

        indices = _breed_indices_for(animal)
        if indices:
            scored = sorted(
                [(probs[i].item(), imagenet_labels[i]) for i in indices],
                reverse=True
            )
            # at least 1 % confidence
            scored = [(p, n) for p, n in scored if p >= 0.005]
            if scored:
                top_name = scored[0][1].split(',')[0].strip()
                top_conf = round(scored[0][0] * 100, 1)
                top5 = [
                    {"breed": n.split(',')[0].strip(), "confidence": round(p * 100, 1)}
                    for p, n in scored[:5]
                ]
                return {"breed": top_name, "breed_confidence": top_conf, "top_breeds": top5}

        # fallback: global top prediction
        top_idx = int(probs.argmax())
        return {
            "breed": imagenet_labels[top_idx].split(',')[0].strip(),
            "breed_confidence": round(probs[top_idx].item() * 100, 1),
            "top_breeds": [],
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# API endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        if yolo_model is None:
            return JSONResponse({
                "success": False,
                "error": "Model henüz eğitilmedi. Lütfen train_classifier.py çalıştırın.",
            })

        yolo_results = yolo_model(image, verbose=False)

        if not yolo_results:
            return JSONResponse({"success": False, "error": "Sonuç alınamadı."})

        r     = yolo_results[0]
        probs = r.probs
        names = r.names  # {0: 'butterfly', 1: 'cat', ...}

        if probs is None:
            return JSONResponse({"success": False, "error": "Sınıflandırma başarısız."})

        top1_idx  = int(probs.top1)
        top1_conf = float(probs.top1conf)
        animal    = names[top1_idx]

        # All animal probabilities (top-5 from YOLOv8)
        top5_idx  = list(probs.top5)
        top5_conf = probs.top5conf.tolist()
        all_animals = [
            {"animal": names[int(idx)], "animal_tr": ANIMAL_TR.get(names[int(idx)], names[int(idx)]),
             "confidence": round(float(c) * 100, 1)}
            for idx, c in zip(top5_idx, top5_conf)
        ]

        # Breed detection
        breed_info = detect_breed(image, animal) if top1_conf >= 0.25 else None

        return JSONResponse({
            "success":           True,
            "animal":            animal,
            "animal_tr":         ANIMAL_TR.get(animal, animal),
            "animal_confidence": round(top1_conf * 100, 1),
            "emoji":             ANIMAL_EMOJI.get(animal, "🐾"),
            "breed":             breed_info["breed"]           if breed_info else None,
            "breed_confidence":  breed_info["breed_confidence"] if breed_info else None,
            "top_breeds":        breed_info["top_breeds"]      if breed_info else [],
            "all_animals":       all_animals,
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


# ──────────────────────────────────────────────────────────────────────────────
# Web UI
# ──────────────────────────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hayvan & Irk Tanıma AI</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0f1117;--card:#1a1d27;--card2:#22263a;
    --accent:#7c6df7;--accent2:#5eead4;--gold:#fbbf24;
    --text:#e2e8f0;--muted:#94a3b8;
  }
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
       min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:20px}

  h1{font-size:2rem;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:24px 0 4px}
  .subtitle{color:var(--muted);font-size:.95rem;margin-bottom:32px;text-align:center}

  .main{display:grid;grid-template-columns:1fr 1fr;gap:24px;width:100%;max-width:960px}
  @media(max-width:700px){.main{grid-template-columns:1fr}}

  .card{background:var(--card);border-radius:16px;padding:24px;
        border:1px solid rgba(124,109,247,.2)}

  /* Upload zone */
  #dropZone{border:2px dashed rgba(124,109,247,.5);border-radius:12px;padding:36px 20px;
            text-align:center;cursor:pointer;transition:.2s;user-select:none}
  #dropZone:hover,#dropZone.drag{border-color:var(--accent);background:rgba(124,109,247,.07)}
  #dropZone .icon{font-size:3rem;margin-bottom:10px}
  #dropZone p{color:var(--muted);font-size:.9rem;margin-top:6px}
  #fileInput{display:none}

  #previewWrap{margin-top:16px;display:none;text-align:center}
  #previewWrap img{max-width:100%;max-height:280px;border-radius:10px;
                   object-fit:contain;box-shadow:0 4px 20px rgba(0,0,0,.5)}

  .btn{display:block;width:100%;margin-top:16px;padding:12px;border:none;border-radius:10px;
       background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;
       font-size:1rem;font-weight:700;cursor:pointer;transition:.15s;letter-spacing:.3px}
  .btn:hover{opacity:.9;transform:translateY(-1px)}
  .btn:disabled{opacity:.5;cursor:not-allowed;transform:none}

  /* Result panel */
  #resultPanel{display:none}
  .animal-header{display:flex;align-items:center;gap:12px;margin-bottom:20px}
  .animal-emoji{font-size:3.5rem}
  .animal-name{font-size:1.8rem;font-weight:800}
  .animal-name-tr{font-size:1rem;color:var(--muted)}
  .conf-chip{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.8rem;
             font-weight:700;background:rgba(124,109,247,.2);color:var(--accent);margin-top:4px}

  .section-title{font-size:.75rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:.1em;color:var(--muted);margin:20px 0 10px}

  .breed-box{background:var(--card2);border-radius:10px;padding:14px 16px;margin-bottom:8px}
  .breed-name{font-size:1.05rem;font-weight:700;color:var(--accent2)}
  .breed-conf{font-size:.82rem;color:var(--muted);margin-top:3px}

  /* Progress bars */
  .bar-row{margin-bottom:10px}
  .bar-label{display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:4px}
  .bar-track{background:rgba(255,255,255,.07);border-radius:6px;height:8px;overflow:hidden}
  .bar-fill{height:100%;border-radius:6px;
            background:linear-gradient(90deg,var(--accent),var(--accent2));
            transition:width .6s ease}

  /* Spinner */
  .spinner{display:none;justify-content:center;align-items:center;padding:30px}
  .spinner svg{animation:spin 1s linear infinite;width:40px;height:40px}
  @keyframes spin{to{transform:rotate(360deg)}}

  .error-msg{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.3);
             color:#fca5a5;border-radius:10px;padding:14px;font-size:.9rem;display:none}
  .no-model{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3);
            color:var(--gold);border-radius:10px;padding:14px;font-size:.87rem;line-height:1.6}
</style>
</head>
<body>
<h1>🐾 Hayvan & Irk Tanıma AI</h1>
<p class="subtitle">Bir hayvan resmi yükleyin — tür ve ırk bilgisini saniyeler içinde öğrenin</p>

<div class="main">
  <!-- Sol: Yükleme -->
  <div class="card">
    <div id="dropZone" onclick="document.getElementById('fileInput').click()"
         ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
      <div class="icon">📸</div>
      <strong>Resim seçin veya sürükleyin</strong>
      <p>JPG · PNG · WEBP</p>
    </div>
    <input type="file" id="fileInput" accept="image/*">

    <div id="previewWrap"><img id="previewImg" src="" alt="Önizleme"></div>

    <button class="btn" id="analyzeBtn" onclick="analyze()" disabled>Analiz Et</button>
  </div>

  <!-- Sağ: Sonuçlar -->
  <div class="card">
    <!-- Model yüklü değilse uyarı -->
    <div id="noModelMsg" class="no-model" style="display:none">
      ⚠️ <strong>Model henüz eğitilmedi.</strong><br>
      Lütfen sırasıyla şu komutları çalıştırın:<br><br>
      <code>python prepare_dataset.py</code><br>
      <code>python train_classifier.py</code><br>
      <code>python app.py</code>
    </div>

    <div class="spinner" id="spinner">
      <svg viewBox="0 0 50 50"><circle cx="25" cy="25" r="20" fill="none"
        stroke="url(#g)" stroke-width="4" stroke-linecap="round"
        stroke-dasharray="90 30"/><defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#7c6df7"/><stop offset="100%" stop-color="#5eead4"/>
      </linearGradient></defs></svg>
    </div>

    <div class="error-msg" id="errorMsg"></div>

    <div id="resultPanel">
      <!-- Hayvan başlığı -->
      <div class="animal-header">
        <div class="animal-emoji" id="animalEmoji"></div>
        <div>
          <div class="animal-name" id="animalName"></div>
          <div class="animal-name-tr" id="animalNameTr"></div>
          <span class="conf-chip" id="animalConf"></span>
        </div>
      </div>

      <!-- Irk -->
      <div id="breedSection">
        <div class="section-title">Tespit Edilen Irk / Tür</div>
        <div class="breed-box">
          <div class="breed-name" id="breedName"></div>
          <div class="breed-conf" id="breedConf"></div>
        </div>
        <div id="topBreeds"></div>
      </div>

      <!-- Hayvan olasılıkları -->
      <div class="section-title">Hayvan Sınıfı Olasılıkları</div>
      <div id="animalBars"></div>
    </div>
  </div>
</div>

<script>
let selectedFile = null;

// Drag & drop
function onDragOver(e){e.preventDefault();e.currentTarget.classList.add('drag')}
function onDragLeave(e){e.currentTarget.classList.remove('drag')}
function onDrop(e){
  e.preventDefault();e.currentTarget.classList.remove('drag');
  const f=e.dataTransfer.files[0];
  if(f && f.type.startsWith('image/')) loadFile(f);
}

document.getElementById('fileInput').addEventListener('change',e=>{
  if(e.target.files[0]) loadFile(e.target.files[0]);
});

function loadFile(file){
  selectedFile=file;
  const reader=new FileReader();
  reader.onload=ev=>{
    document.getElementById('previewImg').src=ev.target.result;
    document.getElementById('previewWrap').style.display='block';
  };
  reader.readAsDataURL(file);
  document.getElementById('analyzeBtn').disabled=false;
  document.getElementById('resultPanel').style.display='none';
  document.getElementById('errorMsg').style.display='none';
  document.getElementById('noModelMsg').style.display='none';
}

async function analyze(){
  if(!selectedFile) return;
  const btn=document.getElementById('analyzeBtn');
  btn.disabled=true;
  document.getElementById('spinner').style.display='flex';
  document.getElementById('resultPanel').style.display='none';
  document.getElementById('errorMsg').style.display='none';
  document.getElementById('noModelMsg').style.display='none';

  const fd=new FormData();
  fd.append('file',selectedFile);
  try{
    const res=await fetch('/predict',{method:'POST',body:fd});
    const d=await res.json();
    document.getElementById('spinner').style.display='none';
    if(!d.success){
      if(d.error && d.error.includes('eğitilmedi')){
        document.getElementById('noModelMsg').style.display='block';
      } else {
        showError(d.error||'Bilinmeyen hata');
      }
      btn.disabled=false; return;
    }
    showResult(d);
  } catch(err){
    document.getElementById('spinner').style.display='none';
    showError('Sunucuya bağlanılamadı: '+err.message);
  }
  btn.disabled=false;
}

function showError(msg){
  const el=document.getElementById('errorMsg');
  el.textContent='❌ '+msg; el.style.display='block';
}

function showResult(d){
  document.getElementById('animalEmoji').textContent=d.emoji||'🐾';
  document.getElementById('animalName').textContent=d.animal;
  document.getElementById('animalNameTr').textContent=d.animal_tr||'';
  document.getElementById('animalConf').textContent='%'+d.animal_confidence+' güven';

  // Irk
  const breedSec=document.getElementById('breedSection');
  if(d.breed){
    breedSec.style.display='block';
    document.getElementById('breedName').textContent=d.breed;
    document.getElementById('breedConf').textContent=
      d.breed_confidence ? 'Güven: %'+d.breed_confidence : '';

    const topEl=document.getElementById('topBreeds');
    topEl.innerHTML='';
    if(d.top_breeds && d.top_breeds.length>1){
      topEl.innerHTML='<div class="section-title" style="margin-top:12px">Diğer Olası Irklar</div>';
      d.top_breeds.slice(1).forEach(b=>{
        topEl.innerHTML+=`
          <div class="bar-row">
            <div class="bar-label"><span>${b.breed}</span><span>%${b.confidence}</span></div>
            <div class="bar-track"><div class="bar-fill" style="width:${b.confidence}%"></div></div>
          </div>`;
      });
    }
  } else {
    breedSec.style.display='none';
  }

  // Hayvan bar chart
  const barsEl=document.getElementById('animalBars');
  barsEl.innerHTML='';
  const animalEmojis={'butterfly':'🦋','cat':'🐈','chicken':'🐔','cow':'🐄',
                      'dog':'🐕','horse':'🐎','sheep':'🐑'};
  (d.all_animals||[]).forEach(a=>{
    const e=animalEmojis[a.animal]||'🐾';
    barsEl.innerHTML+=`
      <div class="bar-row">
        <div class="bar-label">
          <span>${e} ${a.animal_tr||a.animal}</span>
          <span>%${a.confidence}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${a.confidence}%"></div>
        </div>
      </div>`;
  });

  document.getElementById('resultPanel').style.display='block';
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
