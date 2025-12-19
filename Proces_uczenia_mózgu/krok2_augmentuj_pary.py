import cv2
import os
import glob
import random

# --- Konfiguracja ---
FOLDER_WE_MAPY = "modele"
FOLDER_WE_SZKICE = "szkice"

FOLDER_WY_MAPY = "modele_finalne"
FOLDER_WY_SZKICE = "szkice_finalne"

# 1 oryginał + 3 obroty (90, 180, 270) = 4 -> 4 x 2 (oryginał + odbicie lustrzane) = 8
# Łącznie z 1 kafla -> 8 par
LICZBA_AUGMENTACJI_NA_OBRAZ = 8
# w moim przypadku 61 * 8 = 488 obrazów końcowych -> uczenie modelu na podstawie 448 obrazów zamiast 61

def augmentuj_calosc(mapa, szkic):
    pary_wynikowe = []
    
    # zapisz oryginał
    pary_wynikowe.append((mapa, szkic))
    
    # obroty
    mapa_rot90 = cv2.rotate(mapa, cv2.ROTATE_90_CLOCKWISE)
    szkic_rot90 = cv2.rotate(szkic, cv2.ROTATE_90_CLOCKWISE)
    pary_wynikowe.append((mapa_rot90, szkic_rot90))
    
    mapa_rot180 = cv2.rotate(mapa, cv2.ROTATE_180)
    szkic_rot180 = cv2.rotate(szkic, cv2.ROTATE_180)
    pary_wynikowe.append((mapa_rot180, szkic_rot180))
    
    mapa_rot270 = cv2.rotate(mapa, cv2.ROTATE_90_COUNTERCLOCKWISE)
    szkic_rot270 = cv2.rotate(szkic, cv2.ROTATE_90_COUNTERCLOCKWISE)
    pary_wynikowe.append((mapa_rot270, szkic_rot270))
    
    pary_przed_odwroceniem = pary_wynikowe.copy()
    for m, s in pary_przed_odwroceniem:
        mapa_flip = cv2.flip(m, 1)
        szkic_flip = cv2.flip(s, 1)
        pary_wynikowe.append((mapa_flip, szkic_flip))
    
    return pary_wynikowe


for folder in [FOLDER_WY_MAPY, FOLDER_WY_SZKICE]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Utworzono folder: '{folder}'")

if not os.path.exists(FOLDER_WE_MAPY) or not os.path.exists(FOLDER_WE_SZKICE):
    print(f"Foldery '{FOLDER_WE_MAPY}' lub '{FOLDER_WE_SZKICE}' nie istnieją")
    exit()

# szukanie wszystkich plików .png w folderze wej
pliki_map_oryg = glob.glob(os.path.join(FOLDER_WE_MAPY, "*.png"))
print(f"Znaleziono {len(pliki_map_oryg)} par (mapa/szkic) do augmentacji")

if len(pliki_map_oryg) == 0:
    print("Folder wejściowy jest pusty")
    exit()

globalny_licznik = 0

for sciezka_mapy in pliki_map_oryg:
    nazwa_pliku = os.path.basename(sciezka_mapy)
    sciezka_szkicu = os.path.join(FOLDER_WE_SZKICE, nazwa_pliku)
    
    if not os.path.exists(sciezka_szkicu):
        print(f"Brak szkicu dla {nazwa_pliku}")
        continue
        
    mapa = cv2.imread(sciezka_mapy, cv2.IMREAD_GRAYSCALE)
    szkic = cv2.imread(sciezka_szkicu) # kolor więc bez IMREAD_GRAYSCALE
    
    if mapa is None or szkic is None:
        print(f"Błąd odczytu {nazwa_pliku}")
        continue

    zaaugmentowane_pary = augmentuj_calosc(mapa, szkic)
    
    nazwa_bazy = os.path.splitext(nazwa_pliku)[0]
    
    for i, (nowa_mapa, nowy_szkic) in enumerate(zaaugmentowane_pary):
        nowa_nazwa = f"{nazwa_bazy}_aug_{i}.png"
        
        sciezka_wy_mapy = os.path.join(FOLDER_WY_MAPY, nowa_nazwa)
        cv2.imwrite(sciezka_wy_mapy, nowa_mapa)
        
        sciezka_wy_szkicu = os.path.join(FOLDER_WY_SZKICE, nowa_nazwa)
        cv2.imwrite(sciezka_wy_szkicu, nowy_szkic)
        
        globalny_licznik += 1

print(f"Augmentacja zakończona. Utworzono {globalny_licznik} par.")
print(f"Dane treningowe w folderach: '{FOLDER_WY_MAPY}' i '{FOLDER_WY_SZKICE}'")