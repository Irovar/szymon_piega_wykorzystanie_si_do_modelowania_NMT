import rasterio
import cv2
import numpy as np
import glob
import os


FOLDER_WYJSCIOWY = "mapy_png" 

def konwertuj_tiff_do_png():
    if not os.path.exists(FOLDER_WYJSCIOWY):
        os.makedirs(FOLDER_WYJSCIOWY)
        print(f"Utworzono folder na wyniki: '{FOLDER_WYJSCIOWY}'")
    pliki_tiff = glob.glob('*.tif*')
    
    if not pliki_tiff:
        print("Nie znaleziono żadnych plików .tif ani .tiff w tym folderze.")
        return

    print(f"Znaleziono {len(pliki_tiff)} plików TIFF do przetworzenia...")
    licznik = 1

    for nazwa_pliku in pliki_tiff:
        print(f"\n--- Przetwarzam plik: {nazwa_pliku} ---")
        
        try:
            with rasterio.open(nazwa_pliku) as src:
                dane_wysokosci = src.read(1).astype(np.float32)
                no_data_value = src.nodata

                if no_data_value is not None:
                    dane_poprawne = dane_wysokosci[dane_wysokosci != no_data_value]
                    if dane_poprawne.size > 0:
                        rzeczywisty_min = dane_poprawne.min()
                    else:
                        rzeczywisty_min = 0
                    
                    dane_wysokosci[dane_wysokosci == no_data_value] = rzeczywisty_min
                
                mapa_8bit = cv2.normalize(
                    src=dane_wysokosci, 
                    dst=None, 
                    alpha=0, 
                    beta=255, 
                    norm_type=cv2.NORM_MINMAX, 
                    dtype=cv2.CV_8U  
                )
                nazwa_wyjsciowa = f"mapa_wysokosci_{licznik:03d}.png"
                sciezka_wyjsciowa = os.path.join(FOLDER_WYJSCIOWY, nazwa_wyjsciowa)

                cv2.imwrite(sciezka_wyjsciowa, mapa_8bit)
                print(f"Zapisano jako: {sciezka_wyjsciowa}")

                licznik += 1

        except Exception as e:
            print(f"Błąd podczas przetwarzania pliku {nazwa_pliku}: {e}")
            print("Pominięto ten plik.")

    print(f"\nPrzetworzono i zapisano {licznik - 1} plików w folderze '{FOLDER_WYJSCIOWY}'.")

if __name__ == "__main__":
    konwertuj_tiff_do_png()