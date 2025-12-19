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
```

### 2. Środowisko Windows (Wizualizacja 3D)
Plik `OpenGL.exe` jest skompilowany statycznie i nie wymaga instalacji środowiska programistycznego.

## Sterowanie w oknie 3D

Po uruchomieniu wizualizacji sterowanie kamerą odbywa się następująco:

| Akcja | Klawisz / Mysz |
| :--- | :--- |
| **Obrót kamery** | Przytrzymaj **Lewy Przycisk Myszy** i ruszaj myszką |
| **Przesuwanie mapy** | Przytrzymaj **Prawy Przycisk Myszy** i ruszaj myszką |
| **Przybliżanie (Zoom)** | **Scroll** myszy |
| **Zrzut ekranu** | Klawisz **P** (zapisuje plik `screenshot_model.png` w folderze aplikacji) |


## Dostęp do kodu źródłowego

W celu weryfikacji implementacji silnika graficznego, do repozytorium dołączono **pełny katalog projektu Visual Studio**. Pozwala to na podgląd struktury kodu oraz samodzielną kompilację bez konieczności ręcznej konfiguracji bibliotek zewnętrznych.

**Zawartość folderu projektowego:**
* `OpenGL.sln` - Główny plik rozwiązania – otwiera cały projekt w Visual Studio.
* `src/` - Folder z plikami źródłowymi C++.
* `Dependencies/` - Skonfigurowane biblioteki zewnętrzne.
* Pliki konfiguracyjne (`.vcxproj`, `.filters`).

Projekt jest skonfigurowany relatywnie, co oznacza, że po pobraniu repozytorium można go od razu zbudować w środowisku Visual Studio 2022.
