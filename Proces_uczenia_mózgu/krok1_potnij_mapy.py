import cv2
import os
import glob
import numpy as np

# ustawienia
FOLDER_WEJ = "mapy_png"
FOLDER_WYJ = "mapy_uciete"
ROZMIAR_KAFELKA = 512
PRZESUNIECIE = 256 

if not os.path.exists(FOLDER_WEJ):
    print(f"Folder wejściowy '{FOLDER_WEJ}' nie istnieje.")
    exit()

if not os.path.exists(FOLDER_WYJ):
    os.makedirs(FOLDER_WYJ)
    print(f"Utworzono folder wyjściowy: '{FOLDER_WYJ}'")

# szukanie wszystkich plików PNG w folderze wejściowym
pliki_map = glob.glob(os.path.join(FOLDER_WEJ, "*.png"))
print(f"Znaleziono {len(pliki_map)} map do pocięcia...")

licznik = 0

# pętla dla wszystkich znalezionych map z folderu wejściowego
for sciezka_mapy in pliki_map:
    mapa = cv2.imread(sciezka_mapy, cv2.IMREAD_GRAYSCALE)
    if mapa is None:
        print(f"Nie udało się wczytać mapy: {sciezka_mapy}")
        continue
    
    # pobieranie nazwy bazy pliku do tworzenia nazw kafli
    nazwa_bazy = os.path.splitext(os.path.basename(sciezka_mapy))[0]
    
    wysokosc, szerokosc = mapa.shape
    print(f"Przetwarzam: {nazwa_bazy} (rozmiar: {szerokosc}x{wysokosc})")

    # przesunięcie się w dół po y
    for y in range(0, wysokosc - ROZMIAR_KAFELKA + 1, PRZESUNIECIE):
        # przesunięcie się w prawo po x
        for x in range(0, szerokosc - ROZMIAR_KAFELKA + 1, PRZESUNIECIE):
            
            # wycinanie kafla
            kafelek = mapa[y : y + ROZMIAR_KAFELKA, x : x + ROZMIAR_KAFELKA]
            
            # zapis kafla
            nazwa_pliku_wy = f"{nazwa_bazy}_tile_x{x}_y{y}.png"
            sciezka_wy = os.path.join(FOLDER_WYJ, nazwa_pliku_wy)
            
            cv2.imwrite(sciezka_wy, kafelek)
            licznik += 1

print(f"\nPocięto mapy na {licznik} kafli.")
print(f"Wyniki w folderze: '{FOLDER_WYJ}'")