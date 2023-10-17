import tkinter as tk
import numpy as np
import asyncio
import time
import math
UCOL = 255
##np.seterr(divide='ignore')
pc = time.perf_counter
F = 2
def _photo_image(image: np.ndarray):
    height, width = image.shape
    data = f'P5 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')
def _photo_image_col(image: np.ndarray):
    height, width = image.shape[:2]
    data = f'P6 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')
def colorize(image: np.ndarray):
    nimg = image.reshape(image.shape + (1,)).repeat(3, 2)
    nimg = (0.5 + np.sin((nimg / K) ** 0.55 * F * math.pi))
    nimg[:, :, 0] = nimg[:, :, 0] * ((1 - nimg[:, :, 0]) * 0.8 + 0.1) * UCOL
    nimg[:, :, 1] = (nimg[:, :, 1] * 0.75 + np.maximum(0, nimg[:, :, 1] - 0.4) \
                    * 0.3 + nimg[:, :, 1] * (1 - nimg[:, :, 1]) * 0.32) * UCOL
    nimg[:, :, 2] = (nimg[:, :, 2] * 0.75 + np.maximum(0, 0.7 - nimg[:, :, 2]) \
                     * 0.1) * UCOL
##    nimg[:, :, 2] = 0
##    nimg[:, :, 1] = 0
##    nimg[:, :, 1] = 0
    return _photo_image_col(np.array(nimg + 0.5, dtype=np.int32))

c = -0.229 + 0.7j

W = 1920
H = 1000
SOAP = 6
K=205
INF = 3000
A, B, C, D = 1, 1j, 1j, -2
cf = 0.01
cx = cy = 0

NW = int(W // SOAP)
NH = int(H // SOAP)
CW = NW / 2
CH = NH / 2
cf *= SOAP
root = tk.Tk()
canvas = tk.Canvas(root, width=W,height=H)
canvas.pack()
w = None
base = np.zeros((NH, NW), dtype=np.complex128)
for i in range(NH):
    for j in range(NW):
        base[i][j] = j - 1j * i

def cnt(z):
    k = 0
    while k < K and z.__abs__() < INF:
        k += 1
        z = z ** 2 + c
    return k
cntf = np.vectorize(cnt)
def render():
    global w, zxc
    t1 = pc()
    a = base.copy() * cf - CW * cf + cx + 1j * CH * cf - 1j * cy
    
    a = cntf(a)
##    a -= np.amin(a)
    
##    a = a / K
##    a = (a ** 0.75) * 255
    
##    a = (1 / (K - a + 1) ** 0.25) * 255
##    a = 1.5/a

##    a = (np.sin(a / K * math.pi * 4) + 1) * (255/2)

    a = (a / K) * 255
    
    a[np.isnan(a)] = 255
    a = np.minimum(a, 255)
    
##    print(a)
##    for i in range(NH):
##        for j in range(NW):
##            asyncio.ensure_future(rpixel(a, i, j))
##    loop = asyncio.get_event_loop()
##    pending = asyncio.Task.all_tasks()
##    loop.run_until_complete(asyncio.gather(*pending))
    
##    img = _photo_image(a)

    img = colorize(a)
    sw = W // NW
    sh = H // NH
    img=img.zoom(sw, sh)
    canvas.create_image(0, 0, anchor='nw', image=img)
    w = img
            
def ML(e):
    global cx
    cx -= cf * 20
    render()
def MR(e):
    global cx
    cx += cf * 20
    render()
def MU(e):
    global cy
    cy -= cf * 20
    render()
def MD(e):
    global cy
    cy += cf * 20
    render()
def INC(e):
    global cf
    cf *= 1.3
    render()
def DEC(e):
    global cf
    cf /= 1.3
    render()
def NF(e):
    global F
    F *= 1.05
    render()
def PF(e):
    global F
    F /= 1.05
    render()
##frame = tk.Frame(root, width=W, height=70)
##btn = tk.Button(frame, text="1", width=20, height=2, command=render, anchor='nw')
##btn.place(x=50,y=10)
##btn = tk.Button(frame, text="1", width=20, height=2, command=render, anchor='nw')
##btn.place(x=W*0.8-50,y=10)
##frame.pack()
root.bind("<Left>", ML)
root.bind("<Right>", MR)
root.bind("<Up>", MU)
root.bind("<Down>", MD)
root.bind("<q>", DEC)
root.bind("<a>", INC)
root.bind("<w>", NF)
root.bind("<s>", PF)

render()
root.mainloop()
