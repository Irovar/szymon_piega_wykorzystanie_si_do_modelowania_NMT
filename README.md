# Wykorzystanie SI do modelowania NMT

**Autor:** Szymon Piega  
**Typ projektu:** Praca Inżynierska

Projekt łączy algorytmy sztucznej inteligencji (cGAN, pix2pix) zaimplementowane w języku Python z interaktywną wizualizacją terenu 3D napisaną w języku C++ (OpenGL).

## Struktura projektu

Repozytorium zawiera pliki wykonywalne oraz pełny kod źródłowy:
* `Wrapper.py` - Główny skrypt łączący generator i wizualizację. **Zalecany sposób uruchomienia.**
* `OpenGL.exe` - Skompilowany silnik renderujący teren 3D.
* `generator_mozg.pth` - Wyuczony model sieci neuronowej.
* `generator_aplikacja.py` - Skrypt generujący NMT na podstawie wbudowanego szkicownika.
* **Folder z projektem VS** - Pełny projekt Visual Studio (kod źródłowy C++).

## Ważna uwaga: Generowanie terenu
Repozytorium **nie zawiera** domyślnego pliku tekstury terenu (`terrain.png`).
Aby poprawnie uruchomić wizualizację, należy:
1. Uruchomić skrypt `Wrapper.py` – przeprowadzi on użytkownika przez proces generowania terenu i automatycznie uruchomi wizualizację.
2. Alternatywnie: Użyć `generator_aplikacja.py` do stworzenia pliku `terrain.png`, a następnie ręcznie uruchomić `OpenGL.exe`.

## Wymagania i Instalacja (Uruchomienie aplikacji)

### 1. Środowisko Python
Do działania generatora wymagany jest **Python (wersja 3.8 lub nowsza)**.

Wymagane biblioteki:
* PyTorch
* NumPy
* OpenCV (`opencv-python`)
* Pillow

**Instalacja zależności:**
Otwórz terminal w folderze projektu i wpisz:
```bash
pip install torch numpy opencv-python pillow
