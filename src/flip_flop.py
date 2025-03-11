import lip
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np

# Lista de imágenes y títulos para navegación
images = []
image_titles = []
current_index = 0  # Índice de la imagen actual


def convert_to_tk(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return ImageTk.PhotoImage(image)

def show_image(index):
    global current_index, images, image_titles
    current_index = index % len(images)
    tk_image = convert_to_tk(images[current_index])
    img_label.config(image=tk_image)
    img_label.image = tk_image
    title_label.config(text=image_titles[current_index])

def save_image(img):
    global current_index
    filename = f"output_{image_titles[current_index].replace(' ', '_').lower()}.jpg"
    img = lip.get_flip_flop_flipflop(img, image_titles[current_index])
    lip.save(filename, img)
    print(f"Imagen guardada como {filename}")

def main():
    
    path = lip.parse_args_path()
    img = lip.open_image(path) 
    
    if img is None:
        print("Error: Could not load the image.")
        exit()

    img_copy = img.copy()
    img_copy = lip.resize_image(img_copy, 1280, 720)

    images.append(img_copy)
    images.extend(lip.flip_flop_flipflop(img_copy))
    
    image_titles.extend(["original image", "Flip", "Flop", "Flip-Flop"])

    # Configura la ventana principal
    root = tk.Tk()
    root.title("Image Navigator")

    global img_label, title_label

    # Contenedor para la imagen y título
    img_label = Label(root)
    img_label.pack()

    title_label = Label(root, text="", font=("Arial", 16))
    title_label.pack()

    # Botones de navegación
    btn_prev = tk.Button(root, text="Previous", command=lambda: show_image(current_index - 1))
    btn_prev.pack(side="left", padx=10)

    btn_next = tk.Button(root, text="Next", command=lambda: show_image(current_index + 1))
    btn_next.pack(side="right", padx=10)

    btn_save = tk.Button(root, text="Save", command=lambda: save_image(img))
    btn_save.pack(side="bottom", pady=10)

    # Muestra la primera imagen
    show_image(current_index)

    # Inicia el loop de la interfaz
    root.mainloop()

if __name__ == "__main__":
    main()
