import subprocess
import os
import sys
import time

def main():
    print("--- START SYSTEMU MODELOWANIA NMT ---")
    
    # Uruchomienie generatora (Python)
    print("[1/2] Uruchamianie modułu generacji terenu...")
    
    # subprocess.run czeka, na zamkniecie okna generatora
    try:
        subprocess.run([sys.executable, "generator_aplikacja.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Błąd generatora: {e}")
        return


    #dodanie mierzenia czasu przejscia, czas kiedy po zamknięciu generatora uruchamiamy wizualizer 3D
    start_transition = time.perf_counter()


    # Sprawdzenie czy plik istnieje
    nazwa_pliku = "terrain.png"
    if not os.path.exists(nazwa_pliku):
        print(f"BŁĄD: Nie znaleziono pliku {nazwa_pliku}. Czy zapisałeś wynik?")
        return

    nazwa_exe = "OpenGL.exe"
    if not os.path.exists(nazwa_exe):
        print(f"BŁĄD: Nie znaleziono pliku {nazwa_exe}!")
        return
    
    end_transition = time.perf_counter()
    czas_przejscia = end_transition - start_transition

    # Uruchomienie wizualizera (C++)
    print("[2/2] Uruchamianie wizualizacji 3D (C++ / OpenGL)...")
    subprocess.run([nazwa_exe])

    print(f"Czas przeładowania danych: {czas_przejscia:.8f} s")
    print("--- KONIEC PRACY SYSTEMU ---")

if __name__ == "__main__":
    main()