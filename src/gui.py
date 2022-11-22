# import webbrowser
import sys
import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import cv2
import time
import timeit
import PIL.ImageTk, PIL.Image
from eigenface import *
from extractor import *

# Path to asset files for this GUI window.
ASSETS_PATH = Path(__file__).resolve().parent / "gui"

# Required in order to add data files to Windows executable
path = getattr(sys, '_MEIPASS', os.getcwd())
os.chdir(path)

output_path = ""

eigenfaces = mean = weight_Train = list_Nama = list_foto =  None

def cv2ImgtoPhoto(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    photo1 = PIL.Image.fromarray(img)
    photo = PIL.ImageTk.PhotoImage(image=photo1)
    return photo
    
def test_button_do():
    image = file_input.get()
    if not image:
        tk.messagebox.showerror(
            title="Empty Fields!", message="Please enter your image test!")
        return
    if not list_foto:
        tk.messagebox.showerror(
            title="Error!", message="No training data!")
        return
    
    img_Test = extractImg(image)
    face_Test = getFaceImage(img_Test)[0]
    isError = False

    if face_Test is not None:
       
        start = timeit.default_timer()
        closest_idx, dist = test(face_Test,eigenfaces,mean,weight_Train)
        end = timeit.default_timer()
        execution_time = end-start
    else :
        isError = True
        error_msg = "Wajah tidak ditemukan pada gambar test!"
    
    if not isError:
        result = cv2ImgtoPhoto(list_foto[closest_idx])
        photo_Test = cv2ImgtoPhoto(img_Test)
        img1 = canvas.itemconfig(image_1, image=photo_Test)
        img2 = canvas.itemconfig(image_2, image=result)
        canvas.itemconfig(info_text, text=f"Gambar Ditemukan! Dist: {dist}")
        canvas.itemconfig(exec_time_text, text=f"Execution Time: {execution_time:.3f} s")
        canvas.tag_raise(img1)
        canvas.tag_raise(img2)
    else :
        tk.messagebox.showerror(
            title="Error!", message=error_msg)
        return

    return

def train_button_do():
    global eigenfaces, mean, weight_Train, list_Nama, list_foto
    folder = folder_input.get()
    if not folder:
        tk.messagebox.showerror(
            title="Empty Fields!", message="Please enter folder.")
        return
    isError = False
    start = timeit.default_timer()
    list_foto, data_set, list_Nama = data_extractor(folder)
    if list_foto:
        eigenfaces, mean, weight_Train = train(data_set)
        end = timeit.default_timer()
        execution_time = end-start
    else:
        isError = True
        error_msg = "Tidak ada wajah pada dataset!"

    if not isError:
        canvas.itemconfig(info_text, text=f"Training selesai!")
        canvas.itemconfig(exec_time_text, text=f"Execution Time: {execution_time:.3f} s")
    else :
        tk.messagebox.showerror(
            title="Error!", message=error_msg)
    return

def select_path():
    global output_path

    output_path = tk.filedialog.askdirectory()
    folder_input.delete(0, tk.END)
    folder_input.insert(0, output_path)

def select_path_input():
    global output_path

    output_path = tk.filedialog.askopenfilename()
    file_input.delete(0, tk.END)
    file_input.insert(0, output_path)

def make_label(master, x, y, h, w, *args, **kwargs):
    f = tk.Frame(master, height=h, width=w)
    f.pack_propagate(0) 
    f.place(x=x, y=y)

    label = tk.Label(f, *args, **kwargs)
    label.pack(fill=tk.BOTH, expand=1)

    return label

def resizeImage(img, newWidth, newHeight):
    oldWidth = img.width()
    oldHeight = img.height()
    newPhotoImage = tk.PhotoImage(width=newWidth, height=newHeight)
    for x in range(newWidth):
        for y in range(newHeight):
            xOld = int(x*oldWidth/newWidth)
            yOld = int(y*oldHeight/newHeight)
            rgb = '#%02x%02x%02x' % img.get(xOld, yOld)
            newPhotoImage.put(rgb, (x, y))
    return newPhotoImage

def open_cam():
    if eigenfaces is None :
        tk.messagebox.showerror(
            title="Error!", message="Masukkan data train terlebih dahulu!")
        return
    global canvas
    for i in range(10):
        testcap = cv2.VideoCapture(i)
        ret, img = testcap.read()
        if ret:
            testcap.release()
            break
    cap = cv2.VideoCapture(i)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
    while True:
        ret, img = cap.read()
        face, x, y, w, h = getFaceImage(img)
        if face is not None:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            closest_idx, dist = test(face,eigenfaces,mean,weight_Train)
            result = cv2ImgtoPhoto(list_foto[closest_idx])
            photo_Test = cv2ImgtoPhoto(face)
            tmp1 = canvas.itemconfig(image_1, image=photo_Test)
            tmp2 = canvas.itemconfig(image_2, image=result)
            canvas.itemconfig(info_text, text=f"Dist: {dist}")
            canvas.tag_raise(tmp1)
            canvas.tag_raise(tmp2)
        cv2.imshow('a', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


window = tk.Tk()
window.title("Face Recognition")

window.geometry("1000x480")
window.configure(bg="#005AC3")
canvas = tk.Canvas(
    window, bg="#005AC3", height=480, width=1000,
    bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)
canvas.create_rectangle(600, 0, 1000, 480, fill="white", outline="")
canvas.create_rectangle(668, 80, 668 + 260, 80 + 3, fill="#005AC3", outline="")

title = tk.Label(
    text="EigenFace Recognition", bg="white",
    fg="black", font=("Arial-BoldMT", int(20.0)))
title.place(x=655.0, y=40.0)

text_box_bg = tk.PhotoImage(file=ASSETS_PATH / "TextBox_Bg.png")
input_image_img = canvas.create_image(800.5, 167.5, image=text_box_bg)
input_dataset_img = canvas.create_image(800.5, 248.5, image=text_box_bg)

file_input = tk.Entry(bd=0, bg="#F6F7F9",fg="#000716",  highlightthickness=0)
file_input.place(x=640.0, y=137+25, width=321.0, height=35)
file_input.focus()


folder_input = tk.Entry(bd=0, bg="#F6F7F9", fg="#000716",  highlightthickness=0)
folder_input.place(x=640.0, y=218+25, width=321.0, height=35)

path_picker_img = tk.PhotoImage(file = ASSETS_PATH / "path_picker.png")
path_folder = tk.Button(
    image = path_picker_img,
    text = '',
    compound = 'center',
    fg = 'white',
    borderwidth = 0,
    highlightthickness = 0,
    command = select_path,
    relief = 'flat')
path_folder.place(
    x = 940, y = 225,
    width = 24,
    height = 22)

path_input = tk.Button(
    image = path_picker_img,
    text = '',
    compound = 'center',
    fg = 'white',
    borderwidth = 0,
    highlightthickness = 0,
    command = select_path_input,
    relief = 'flat')
path_input.place(
    x = 940, y = 145,
    width = 24,
    height = 22)

camera_img = tk.PhotoImage(file = ASSETS_PATH / "cam.png")
camera = tk.Button(
    image = camera_img,
    text = '',
    compound = 'center',
    fg = 'white',
    borderwidth = 0,
    highlightthickness = 0,
    command = open_cam,
    relief = 'flat')
camera.place(
    x = 900, y = 145,
    width = 24,
    height = 22)

canvas.create_text(
    645.0, 150.0, text="Insert Your Photo Directory", fill="#515486",
    font=("Arial-BoldMT", int(13.0)), anchor="w")
canvas.create_text(
    645.0, 230.5, text="Insert Your Dataset Directory", fill="#515486",
    font=("Arial-BoldMT", int(13.0)), anchor="w")
canvas.create_text(
    840.5, 428.5, text="Generate",
    fill="#FFFFFF", font=("Arial-BoldMT", int(13.0)))


canvas.create_text(
    118.0,
    81.0,
    anchor="nw",
    text="Test Image",
    fill="#ffffff",
    font=("Georgia", 15 * -1)
)

canvas.create_text(
    394.0,
    81.0,
    anchor="nw",
    text="Closest Result",
    fill="#ffffff",
    font=("Georgia", 15 * -1)
)

default = tk.PhotoImage(file = ASSETS_PATH / "image_1.png")
default_resized = resizeImage(default, 256, 256)
image_1 = canvas.create_image(
    156.0,
    240.0,
    image=default_resized
)

default2 = tk.PhotoImage(file = ASSETS_PATH / "image_1.png")
default2_resized = resizeImage(default, 256, 256)
image_2 = canvas.create_image(
    440.0,
    240.0,
    image=default2_resized
)

exec_time_text = canvas.create_text(
        717.0, 418.0, text=f"Execution Time: 0.000 s", fill="#515486",
            font=("Aril-BoldMT", int(13.0)), anchor="w")

info_text = canvas.create_text(
            700.0, 390.0, text=" ", fill="#515486",
            font=("Aril-BoldMT", int(13.0)), anchor="w")

train_btn_img = tk.PhotoImage(file=ASSETS_PATH / "train.png")
train_btn = tk.Button(
    image=train_btn_img, borderwidth=0, highlightthickness=0,
    command= train_button_do, relief="flat")
train_btn.place(x=616, y=301, width=127, height=40)

test_btn_img = tk.PhotoImage(file=ASSETS_PATH / "test.png")
test_btn = tk.Button(
    image=test_btn_img, borderwidth=0, highlightthickness=0,
    command=test_button_do, relief="flat")
test_btn.place(x=816, y=301, width=127, height=40)

window.resizable(False, False)
window.mainloop()