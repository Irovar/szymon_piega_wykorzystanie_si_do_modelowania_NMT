import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import torch
import torch.nn as nn
import numpy as np
import cv2
import os

print(f"PyTorch wersja: {torch.__version__}")
# Automatycznie wykryj GPU, jeśli jest, w przeciwnym razie użyj CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używane urządzenie: {DEVICE}")

# --- 1. DEFINICJA MODELU GENERATORA (U-Net) ---
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

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super().__init__()
        # (Encoder)
        self.down1 = BlokGeneratora(in_channels, features, down=True, act="leaky", use_dropout=False)
        self.down2 = BlokGeneratora(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down3 = BlokGeneratora(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down4 = BlokGeneratora(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = BlokGeneratora(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = BlokGeneratora(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down7 = BlokGeneratora(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"), nn.ReLU()
        )
        # (Decoder)
        self.up1 = BlokGeneratora(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = BlokGeneratora(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = BlokGeneratora(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = BlokGeneratora(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = BlokGeneratora(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = BlokGeneratora(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = BlokGeneratora(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

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

# --- 2. KLASA APLIKACJI (GUI) ---
class DrawingApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.model.eval()
        
        self.root.title("Generator Terenu AI")

        # Parametry rysowania
        self.canvas_width = 512
        self.canvas_height = 512
        self.draw_color = "red"
        self.line_width = 5
        self.last_x, self.last_y = None, None
        self.pil_image = Image.new("RGB", (self.canvas_width, self.canvas_height), "black")
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self.setup_gui()

    def setup_gui(self):
        # Płótno
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="black", cursor="cross")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        # Ramka
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Przyciski
        tk.Label(button_frame, text="Narzędzia:", font=("Arial", 12, "bold")).pack(pady=5)

        tk.Label(button_frame, text="Grubość pędzla:").pack(pady=(5, 0))
        self.width_scale = tk.Scale(button_frame, from_=4, to=10, orient=tk.HORIZONTAL, command=self.update_line_width)
        self.width_scale.set(self.line_width)
        self.width_scale.pack(fill=tk.X, pady=5)

        tk.Button(button_frame, text="Rysuj Czerwonym (Wyżyny)", command=lambda: self.set_color("red")).pack(fill=tk.X, pady=5)
        tk.Button(button_frame, text="Rysuj Niebieskim (Niziny)", command=lambda: self.set_color("blue")).pack(fill=tk.X, pady=5)
        tk.Button(button_frame, text="Gumka (Czarne tło)", command=lambda: self.set_color("black")).pack(fill=tk.X, pady=5)
        
        tk.Frame(button_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=10) # Separator

        tk.Button(button_frame, text="GENERUJ MAPĘ", font=("Arial", 12, "bold"), bg="green", fg="white", command=self.generate_map).pack(fill=tk.X, pady=10)
        tk.Button(button_frame, text="Wyczyść wszystko", command=self.clear_canvas).pack(fill=tk.X, pady=5)

        # Powiązanie myszy z funkcjami
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def update_line_width(self, val):
        """Aktualizuje grubość linii na podstawie wartości z suwaka."""
        self.line_width = int(val)
        
    def set_color(self, new_color):
        self.draw_color = new_color

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.last_x and self.last_y:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                     fill=self.draw_color, width=self.line_width, 
                                     capstyle=tk.ROUND, smooth=tk.TRUE)
            
            self.pil_draw.line([self.last_x, self.last_y, x, y], 
                               fill=self.draw_color, width=self.line_width, 
                               joint="curve")

            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="black")

    def generate_map(self):
        print("Rozpoczynam generowanie...")
        try:
            szkic_rgb = np.array(self.pil_image)
            
            # trening byl na BGR, a obraz z PIL jest w RGB
            szkic_bgr = cv2.cvtColor(szkic_rgb, cv2.COLOR_RGB2BGR)
            szkic_bgr_fixed = szkic_bgr.copy()
            szkic_bgr_fixed[:, :, 0] = szkic_bgr[:, :, 2]
            szkic_bgr_fixed[:, :, 2] = szkic_bgr[:, :, 0]
            
            # Normalizacja do [-1, 1]
            szkic_norm = (szkic_bgr_fixed.astype(np.float32) / 127.5) - 1.0
            
            # Wymiary (H, W, C) -> (C, H, W) i wymiar partii
            szkic_tensor = torch.from_numpy(szkic_norm.transpose(2, 0, 1))
            szkic_tensor = szkic_tensor.unsqueeze(0).to(DEVICE)

            # Inferencja modelu
            with torch.no_grad():
                mapa_wyjsciowa_tensor = self.model(szkic_tensor)

            # Postprocessing
            mapa_wynik = mapa_wyjsciowa_tensor.squeeze(0).squeeze(0).cpu()
            mapa_denorm = (mapa_wynik.numpy() + 1.0) * 127.5
            mapa_finalna_img = mapa_denorm.astype(np.uint8)

            # 4. Zapiszanie i wyświetlanie wyniku
            nazwa_pliku_wynik = "terrain.png"
            cv2.imwrite(nazwa_pliku_wynik, mapa_finalna_img)
            
            print(f"Sukces! Mapa zapisana jako {nazwa_pliku_wynik}")
            messagebox.showinfo("Sukces!", f"Mapa została wygenerowana i zapisana jako:\n{nazwa_pliku_wynik}")
            
            try:
                os.startfile(nazwa_pliku_wynik)
            except:
                pass
            
        except Exception as e:
            print(f"Błąd podczas generowania: {e}")
            messagebox.showerror("Błąd", f"Wystąpił błąd:\n{e}")


def main():
    NAZWA_PLIKU_MODELU = "generator_mozg.pth"
    
    if not os.path.exists(NAZWA_PLIKU_MODELU):
        messagebox.showerror("Błąd krytyczny", 
                             f"Nie znaleziono pliku modelu: {NAZWA_PLIKU_MODELU}\n"
                             f"Upewnij się, że plik 'mózgu' jest w tym samym folderze.")
        return

    print(f"Ładuję 'mózg' ({NAZWA_PLIKU_MODELU}) na {DEVICE}...")
    try:
        model = Generator(in_channels=3, out_channels=1).to(DEVICE)
        model.load_state_dict(torch.load(NAZWA_PLIKU_MODELU, map_location=DEVICE))
        print("Mózg załadowany pomyślnie.")
    except Exception as e:
        messagebox.showerror("Błąd ładowania modelu", f"Nie można załadować modelu:\n{e}")
        return

    root = tk.Tk()
    app = DrawingApp(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()