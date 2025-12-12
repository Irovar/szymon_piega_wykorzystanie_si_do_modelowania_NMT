import subprocess
import os
import sys
import time

def main():
    print("--- START SYSTEMU MODELOWANIA NMT ---")
    
    # KROK 1: Uruchomienie generatora (Python)
    print("[1/2] Uruchamianie modułu generacji terenu...")
    
    # subprocess.run czeka, na zamkniecie okna generatora
    try:
        subprocess.run([sys.executable, "generator_aplikacja.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Błąd generatora: {e}")
        return

    # KROK 2: Sprawdzenie czy plik istnieje
    nazwa_pliku = "terrain.png"
    if not os.path.exists(nazwa_pliku):
        print(f"BŁĄD: Nie znaleziono pliku {nazwa_pliku}. Czy zapisałeś wynik?")
        return
    else:
        print(f"Znaleziono nowy model terenu: {nazwa_pliku}")

    # KROK 3: Uruchomienie wizualizera (C++)
    print("[2/2] Uruchamianie wizualizacji 3D (C++ / OpenGL)...")
    
    # Tutaj wpisz dokładną nazwę swojego pliku exe
    nazwa_exe = "OpenGL.exe" 
    
    if os.path.exists(nazwa_exe):
        subprocess.run([nazwa_exe])
    else:
        print(f"BŁĄD: Nie znaleziono pliku {nazwa_exe} w tym folderze!")
        print("Upewnij się, że skopiowałeś plik .exe z folderu Visual Studio.")

    print("--- KONIEC PRACY SYSTEMU ---")

if __name__ == "__main__":
    main()