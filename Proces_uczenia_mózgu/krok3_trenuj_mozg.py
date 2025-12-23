import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from torchvision.utils import save_image

import glob
import os
import cv2
import numpy as np
from tqdm import tqdm # daje pasek postępu

print(f"PyTorch wersja: {torch.__version__}")
# sprawdzanie czy jest gpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używane urządzenie: {DEVICE}")

# konfiguracja treningu i danych

class Config:
    FOLDER_MAPY = "modele_finalne"
    FOLDER_SZKICE = "szkice_finalne"
    
    # parametry
    WEJSCIE_KANALY = 3    # szkic jest kolorowy (B,G,R)
    WYJSCIE_KANALY = 1   # mapa wysokości jest czarno-biała
    LICZBA_EPOK = 50      # ile razy przejść przez cały zbiór danych -> więcej = lepiej, ale wolniej, można też ponownie włączyć trening później
    WIELKOSC_PACZKI = 4   # ile obrazków na raz uczyć -> im mniej tym szybciej, ale mniej dokładnie
    SZYBKOSC_NAUKI = 2e-4 # jak duże kroki robić podczas nauki
    BETA1_ADAM = 0.5
    WAGA_L1 = 100         # jak ważna jest wierność pikseli -> im większa tym bardziej szczegółowe mapy, ale mniej realistyczne
    ROZMIAR_OBRAZU = 512  # rozmiar naszych kafli - 512x512 px

cfg = Config()


# to ładuje pary (szkic, mapa) i przygotowuje je do treningu
class MapaSzkicDataset(Dataset):
    def __init__(self, folder_mapy, folder_szkice):
        self.folder_mapy = folder_mapy
        self.folder_szkice = folder_szkice
        self.pliki_szkicow = sorted(glob.glob(os.path.join(folder_szkice, "*.png")))
        print(f"Odnaleziono {len(self.pliki_szkicow)} par obrazów.")

    def __len__(self):
        return len(self.pliki_szkicow)

    def __getitem__(self, index):
        sciezka_szkicu = self.pliki_szkicow[index]
        nazwa_pliku = os.path.basename(sciezka_szkicu)
        sciezka_mapy = os.path.join(self.folder_mapy, nazwa_pliku)

        # wczytanie szkicu (kolorowy, BGR)
        szkic_img = cv2.imread(sciezka_szkicu)
        # wczytanie mapy (czarno-biała)
        mapa_img = cv2.imread(sciezka_mapy, cv2.IMREAD_GRAYSCALE)
        
        #konwersja szkicu z BGR do RGB, czerwony zosatnie czerwonym dla torcha
        szkic_img = cv2.cvtColor(szkic_img, cv2.COLOR_BGR2RGB)

        # szkic (kolor) normalizacja do [-1, 1]:
        szkic_img = (szkic_img.astype(np.float32) / 127.5) - 1.0
        # mapa (czarno-biała) normalizacja do [-1, 1]:
        mapa_img = (mapa_img.astype(np.float32) / 127.5) - 1.0
        
        # zmiana wymiarów dla PyTorch: (H, W, C) -> (C, H, W)
        szkic_tensor = torch.from_numpy(szkic_img.transpose(2, 0, 1))
        # mapa musi mieć dodatkowy wymiar kanału: (H, W) -> (1, H, W)
        mapa_tensor = torch.from_numpy(np.expand_dims(mapa_img, axis=0))

        return szkic_tensor, mapa_tensor

# to służy do budowy bloków generatora
class BlokGeneratora(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

# to służy do generowania mapy ze szkicu
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super().__init__()
        # w dół -> encoder -> obraz jest coraz mniejszy i bardziej skondensowany
        self.down1 = BlokGeneratora(in_channels, features, down=True, act="leaky", use_dropout=False)
        self.down2 = BlokGeneratora(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down3 = BlokGeneratora(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down4 = BlokGeneratora(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = BlokGeneratora(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = BlokGeneratora(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down7 = BlokGeneratora(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"), nn.ReLU()) # tu na końcu nie ma przecinka bo to nie lista
        # w górę -> decoder -> obraz jest coraz większy i mniej skondensowany
        self.up1 = BlokGeneratora(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = BlokGeneratora(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = BlokGeneratora(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = BlokGeneratora(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = BlokGeneratora(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = BlokGeneratora(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = BlokGeneratora(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1), nn.Tanh(), ) # tu na końcu przecinek bo jest lista
        # w down nie ma listy bo to pojedynczy moduł, a w up są konkatenacje więc tam jest lista (konkatenacja czyli łączenie warstw z encoder'a z warstwami z decoder'a)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final_up(torch.cat([u7, d1], 1))

# to służy do budowy bloków dyskryminatora
class BlokDyskryminatora(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.conv(x)

# to służy do odróżniania prawdziwych map od wygenerowanych
class Dyskryminator(nn.Module):
    def __init__(self, in_channels_szkic=3, in_channels_mapa=1):
        super().__init__()
        # dyskryminator dostaje na wejściu szkic i mapę razem
        in_channels_total = in_channels_szkic + in_channels_mapa
        
        self.d1 = BlokDyskryminatora(in_channels_total, 64, stride=2)
        self.d2 = BlokDyskryminatora(64, 128, stride=2)
        self.d3 = BlokDyskryminatora(128, 256, stride=2)
        self.d4 = BlokDyskryminatora(256, 512, stride=1)
        # warstwa końcowa -> prawda/fałsz
        self.final = nn.Conv2d(512, 1, 4, 1, 1, padding_mode="reflect")

    def forward(self, x_szkic, y_mapa):
        # łączymy szkic i mapę wzdłuż wymiaru kanałów
        x = torch.cat([x_szkic, y_mapa], dim=1)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return self.final(x)

# funkcja inicjalizacji wag modeli
# pomaga modelom szybciej się uczyć na podstawie normalnego rozkładu wag
def inicjalizuj_wagi(model, mean=0.0, std=0.02):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, mean, std)

# główna funkcja treningu
def trenuj():
    print("Inicjalizacja modeli")
    # inicjalizowanie modeli
    gen = Generator(in_channels=cfg.WEJSCIE_KANALY, out_channels=cfg.WYJSCIE_KANALY).to(DEVICE)
    disc = Dyskryminator(in_channels_szkic=cfg.WEJSCIE_KANALY, in_channels_mapa=cfg.WYJSCIE_KANALY).to(DEVICE)
    
    # inicjalizowanie wag
    inicjalizuj_wagi(gen)
    inicjalizuj_wagi(disc)
    
    # optymalizatory (jeden dla Generatora, jeden dla Dyskryminatora)
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.SZYBKOSC_NAUKI, betas=(cfg.BETA1_ADAM, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.SZYBKOSC_NAUKI, betas=(cfg.BETA1_ADAM, 0.999))
    
    # strata
    strata_BCE = nn.BCEWithLogitsLoss() # dla dyskryminatora - prawda/fałsz
    strata_L1 = nn.L1Loss()            # dla generatora - wierność pikseli
    
    print("Przygotowanie danych")
    dataset = MapaSzkicDataset(cfg.FOLDER_MAPY, cfg.FOLDER_SZKICE)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.WIELKOSC_PACZKI, 
        shuffle=True, 
        num_workers=4, # zmienna ustalana w zależności od sprzętu
        pin_memory=True
    )
    
    # Ustaw modele w tryb "trening"
    gen.train()
    disc.train()
    
    print(f"Trening (Epok: {cfg.LICZBA_EPOK})...")
    
    for epoch in range(cfg.LICZBA_EPOK):
        loop = tqdm(dataloader, desc=f"Epoka [{epoch+1}/{cfg.LICZBA_EPOK}]")
        
        for idx, (szkic, mapa_prawdziwa) in enumerate(loop):
            szkic = szkic.to(DEVICE)
            mapa_prawdziwa = mapa_prawdziwa.to(DEVICE)

            # trening dyskryminatora na prawdziwych mapach
            mapa_falszywa = gen(szkic) # wygeneruj mapę
            
            D_prawdziwy_out = disc(szkic, mapa_prawdziwa)
            D_prawdziwy_strata = strata_BCE(D_prawdziwy_out, torch.ones_like(D_prawdziwy_out))
            
            # trening dyskryminatora na wygenerowanych mapach
            D_falszywy_out = disc(szkic, mapa_falszywa.detach()) # .detach() zatrzymuje gradienty -> nie uczymy generatora tutaj tylko dyskryminatora
            D_falszywy_strata = strata_BCE(D_falszywy_out, torch.zeros_like(D_falszywy_out))
            
            # łączna strata dyskryminatora
            D_strata = (D_prawdziwy_strata + D_falszywy_strata) / 2
            
            # aktualizuj wagi dyskryminatora
            opt_disc.zero_grad()
            D_strata.backward()
            opt_disc.step()

            # celem generatora (G) jest oszukanie dyskryminatora (D) i bycie blisko oryginału
            # (GAN Loss)
            D_falszywy_out_po_treningu_D = disc(szkic, mapa_falszywa)
            G_GAN_strata = strata_BCE(D_falszywy_out_po_treningu_D, torch.ones_like(D_falszywy_out_po_treningu_D))
            # (L1 Loss) - Jak blisko jest do prawdziwej mapy?
            G_L1_strata = strata_L1(mapa_falszywa, mapa_prawdziwa) * cfg.WAGA_L1
            # Łączna strata Generatora
            G_strata = G_GAN_strata + G_L1_strata
            
            # aktualizacja wag Generatora
            opt_gen.zero_grad()
            G_strata.backward()
            opt_gen.step()

            # aktualizacja postępu
            if idx % 10 == 0:
                loop.set_postfix(D_strata=D_strata.item(), G_strata=G_strata.item(), G_GAN=G_GAN_strata.item(), G_L1=G_L1_strata.item())
        
        # zapis mózgu po każdej epoce -> nie trzeba czekać do końca treningu
        torch.save(gen.state_dict(), "generator_mozg.pth")
        
        szkic_viz = (szkic[0] * 0.5 + 0.5)
        mapa_prawdziwa_viz = (mapa_prawdziwa[0] * 0.5 + 0.5)
        mapa_falszywa_viz = (mapa_falszywa[0] * 0.5 + 0.5)
        
        # muszą być 3 kanały do zapisu jako obraz
        mapa_prawdziwa_viz = mapa_prawdziwa_viz.repeat(3, 1, 1)
        mapa_falszywa_viz = mapa_falszywa_viz.repeat(3, 1, 1)

        # (Szkic | Wygenerowana Mapa | Prawdziwa Mapa)
        save_image(
            torch.cat([szkic_viz, mapa_falszywa_viz, mapa_prawdziwa_viz], dim=2),
            "progress.png"
        )

    print("Trening zakończony")

# --- 6. URUCHOMIENIE ---
if __name__ == "__main__":
    trenuj()