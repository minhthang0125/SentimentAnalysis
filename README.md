# SentimentAnalysis

SentimentAnalysis la ung dung Flask de thu thap binh luan YouTube, lam sach du lieu, phan tich cam xuc bang AI va hien thi ket qua truc quan.

## Tinh nang

- Phan tich URL YouTube
- Phan tich binh luan do nguoi dung tu nhap
- Crawl comment YouTube
- Phan tich cam xuc voi AI va fallback
- Luu ket qua vao SQLite
- Hien thi bieu do va xuat CSV

## Cau truc project

```text
.
|-- app
|   |-- __init__.py
|   `-- routes.py
|-- models
|   `-- sentiment_model.py
|-- static
|   `-- style.css
|-- templates
|   |-- index.html
|   `-- result.html
|-- utils
|   |-- crawler.py
|   `-- text_cleaner.py
|-- main.py
|-- requirements.txt
|-- requirements-ai.txt
`-- README.md
```

## Cai dat

### 1. Tao moi truong ao

```powershell
python -m venv .venv
```

### 2. Kich hoat moi truong

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Cai dependencies

```powershell
pip install -r requirements.txt
pip install -r requirements-ai.txt
```

### 4. Chay ung dung

```powershell
python main.py
```

Mo trinh duyet tai:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

## Database

Ung dung su dung file SQLite:

- `SentimentAnalysis.db`

No luu:

- URL nguon
- Loai nguon
- Backend AI
- Binh luan
- Nhan cam xuc
- Do tin cay

## Ghi chu

- Hien tai URL chi ho tro YouTube
- Neu AI model khong tai duoc, he thong se dung fallback
- Neu khong lay duoc comment, hay kiem tra URL YouTube, ket noi mang va quyen truy cap cong khai cua video

