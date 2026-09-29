# English ASR × Thai MT Benchmark

เครื่องมือ benchmark สำหรับทดสอบการรู้จำเสียงภาษาอังกฤษ (ASR) และการแปล
ภาษาอังกฤษเป็นภาษาไทย (MT) ทั้งแบบแยกส่วนและแบบ pipeline

## ความสามารถ

รันการประเมิน 3 ขั้นตอน โดยแต่ละชุดทดสอบทำซ้ำ 3 รอบ:

1. **Standalone ASR** — ทดสอบ DeepSpeech, Whisper และ Vosk กับไฟล์เสียง
2. **Standalone MT** — ใช้ผลถอดเสียงจาก Whisper เป็นข้อความต้นทาง แล้วทดสอบ
   M2M100, LibreTranslate และ NLLB
3. **Integrated pipeline** — ทดสอบทุกคู่ ASR × MT รวม 9 pipeline

รายงานคะแนน WER/CER สำหรับ ASR และ BLEU/chrF สำหรับ MT พร้อม latency,
real-time factor และการใช้หน่วยความจำ

## โครงสร้างข้อมูล

วางไฟล์ข้อมูลไว้ในโฟลเดอร์ `data/` โดยไฟล์อ้างอิงต้องมีชื่อฐานเดียวกับไฟล์เสียง
(เปลี่ยนนามสกุลเป็น `.txt`) ตัวอย่างเช่น `sample.wav` ต้องมี
`sample.txt` ในทั้งสองโฟลเดอร์อ้างอิง:

```text
data/
├── audio/            # ไฟล์ .wav, .mp3, .ogg หรือ .flac
├── references_asr/   # transcript ภาษาอังกฤษ
└── references_mt/    # คำแปลอ้างอิงภาษาไทย
```

ไฟล์ข้อความอ้างอิงควรเป็น UTF-8 และมีข้อความล้วน ระบบจะอ่านไฟล์เสียงทั้งหมด
ใน `data/audio/` และแจ้งข้อผิดพลาดหากไม่พบไฟล์อ้างอิงที่ตรงกัน

## ติดตั้ง

ใช้ Python และสร้าง virtual environment จากโฟลเดอร์โปรเจกต์:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

บน macOS/Linux ใช้คำสั่งเปิดใช้งาน environment ดังนี้:

```bash
source .venv/bin/activate
```

### เตรียมโมเดล

- **DeepSpeech:** วางไฟล์ `deepspeech-0.9.3-models.pbmm` และ
  `deepspeech-0.9.3-models.scorer` ไว้ที่โฟลเดอร์หลักของโปรเจกต์
- **Vosk:** ดาวน์โหลดโมเดลภาษาอังกฤษ `vosk-model-en-us-0.22` จาก
  [หน้าโมเดล Vosk](https://alphacephei.com/vosk/models) แล้ววางโฟลเดอร์ไว้ที่
  โฟลเดอร์หลัก
- **Whisper, M2M100 และ NLLB:** ไลบรารีจะดาวน์โหลดโมเดลจาก Hugging Face หรือ
  OpenAI เมื่อเรียกใช้ครั้งแรก จึงต้องเชื่อมต่ออินเทอร์เน็ต
- **LibreTranslate:** เป็นบริการแยก ต้องเปิด Docker และให้บริการที่
  `http://localhost:5000/translate` ก่อนเริ่ม benchmark:

  ```bash
  docker run --rm -p 5000:5000 libretranslate/libretranslate
  ```

การตั้งค่า path และพารามิเตอร์ของโมเดลอยู่ใน `config.py` หากต้องการเปลี่ยน
จำนวนรอบทดสอบ ให้แก้ `NUM_RUNS` ใน `controller.py`

## เริ่มใช้งาน

ตรวจสอบว่ามีไฟล์เสียงและไฟล์อ้างอิงครบ แล้วรัน:

```bash
python controller.py
```

โปรแกรมจะสร้างโฟลเดอร์ `results/` และเขียนไฟล์ CSV ต่อไปนี้:

- `asr_standalone.csv` — ผล ASR แต่ละไฟล์และค่าเฉลี่ยต่อโมเดล
- `mt_standalone.csv` — ผล MT แต่ละไฟล์และค่าเฉลี่ยต่อโมเดล
- `pipeline_results.csv` — ผล pipeline รายไฟล์ ค่าเฉลี่ยรายรอบ และค่าเฉลี่ยรวม
- `summary.csv` — สรุปคะแนนเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานต่อ pipeline

การประเมิน pipeline ต้องมีทั้ง transcript ภาษาอังกฤษและคำแปลอ้างอิงภาษาไทย
สำหรับทุกไฟล์เสียง

## เตรียมข้อมูลจาก corpus

หากใช้ corpus ที่มีโฟลเดอร์ `audios/` และไฟล์ `ss-corpus-en.tsv` สามารถใช้
`prepare_data.py` เพื่อคัดลอกไฟล์เสียงและสร้าง transcript ภาษาอังกฤษกับไฟล์
คำแปล placeholder:

```bash
python prepare_data.py
```

โดยค่าเริ่มต้นสคริปต์จะมองหา corpus ในโฟลเดอร์ `sps-corpus-4.0-2026-06-12-en`
ใต้ Desktop หากเก็บไว้ที่อื่น ให้กำหนดตัวแปร `ASR_MT_CORPUS_DIR` เป็น path
ของ corpus ก่อนรัน ตัวแปร `NUM_FILES` ในสคริปต์กำหนดจำนวนไฟล์ที่จะเตรียม

`translate_references.py` ลบแท็ก disfluency/เสียงรบกวนจาก transcript แล้ว
แปลข้อความเป็นภาษาไทยผ่าน Google Translate และเขียนทับไฟล์อ้างอิง MT:

```bash
python translate_references.py
```

สคริปต์นี้ส่งข้อความไปยังบริการแปลภายนอก และแก้ไขไฟล์ใน
`data/references_asr/` และ `data/references_mt/` โดยตรง ควรสำรองข้อมูลและ
ตรวจสอบคำแปลก่อนใช้ประเมินผล

## โครงสร้างโปรเจกต์

```text
config.py          ค่าตั้งค่า path, โมเดล และชื่อไฟล์ผลลัพธ์
input_module.py    โหลดเสียงและข้อความอ้างอิง
asr_module.py      เรียกใช้ DeepSpeech, Whisper และ Vosk
mt_module.py       เรียกใช้ M2M100, LibreTranslate และ NLLB
eval_module.py     คำนวณ WER, CER, BLEU และ chrF
monitor_module.py  วัด latency, RTF, RAM และ VRAM
controller.py      ควบคุมการทดสอบและบันทึกผล CSV
requirements.txt   dependencies ของโปรเจกต์
```

ไฟล์ข้อมูล โมเดลที่ดาวน์โหลด และผล benchmark เป็นไฟล์เฉพาะเครื่องและไม่ได้
รวมอยู่ใน Git repository
