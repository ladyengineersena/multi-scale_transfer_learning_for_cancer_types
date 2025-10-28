# Kurulum ve KullanÄ±m KÄ±lavuzu

## Sistem Gereksinimleri

- Python 3.8 veya Ã¼zeri
- pip (Python paket yÃ¶neticisi)
- Windows, Linux veya macOS iÅŸletim sistemi

## AdÄ±m AdÄ±m Kurulum

### 1. Python Sanal OrtamÄ± OluÅŸturma

`ash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
`

### 2. Gerekli Paketleri YÃ¼kleme

`ash
pip install -r requirements.txt
`

### 3. NLTK Verisini Ä°ndirme

Python terminalinde Ã§alÄ±ÅŸtÄ±rÄ±n:

`python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
`

### 4. TÃ¼rkÃ§e Dil Modeli Ä°ndirme (Opsiyonel)

SpaCy TÃ¼rkÃ§e modeli iÃ§in:

`ash
python -m spacy download tr_core_news_sm
`

## HÄ±zlÄ± BaÅŸlangÄ±Ã§

### Temel KullanÄ±m

`python
from ilac_etkilesimi import IlacEtkilesimTespiti

# Modeli baÅŸlat
tespit = IlacEtkilesimTespiti()

# Metni analiz et
metin = "Aspirin ve warfarin birlikte kullanÄ±ldÄ±ÄŸÄ±nda kanama riski artar."
sonuclar = tespit.analiz_et(metin)

# SonuÃ§larÄ± yazdÄ±r
for sonuc in sonuclar:
    print(f"Ä°laÃ§lar: {', '.join(sonuc['ilaclar'])}")
    print(f"EtkileÅŸim TÃ¼rÃ¼: {sonuc['turu']}")
    print(f"Risk Seviyesi: {sonuc['seviye']}")
`

### Ã–rnekleri Ã‡alÄ±ÅŸtÄ±rma

`ash
cd examples
python ornek_kullanim.py
`

### Testleri Ã‡alÄ±ÅŸtÄ±rma

`ash
python -m pytest tests/
`

## UTF-8 Encoding DesteÄŸi

Proje, TÃ¼rkÃ§e karakterleri (Ã§, ÄŸ, Ä±, Ã¶, ÅŸ, Ã¼) doÄŸru ÅŸekilde iÅŸleyebilir. TÃ¼m dosyalar UTF-8 encoding ile kaydedilmiÅŸtir.

## Sorun Giderme

### TÃ¼rkÃ§e Karakter SorunlarÄ±

EÄŸer TÃ¼rkÃ§e karakterlerde sorun yaÅŸÄ±yorsanÄ±z:

1. IDE veya editÃ¶rÃ¼nÃ¼zÃ¼n encoding ayarÄ±nÄ± kontrol edin (UTF-8 olmalÄ±)
2. Terminal encoding ayarÄ±nÄ± kontrol edin
3. Windows iÃ§in: chcp 65001 komutunu Ã§alÄ±ÅŸtÄ±rÄ±n

### NLTK Ä°ndirme SorunlarÄ±

NLTK paketleri otomatik indirilemiyorsa:

`python
import nltk
nltk.download('punkt', download_dir='C:/nltk_data')
nltk.download('stopwords', download_dir='C:/nltk_data')
`

## Ä°letiÅŸim

SorularÄ±nÄ±z iÃ§in GitHub Issues sayfasÄ±nÄ± kullanÄ±n.
