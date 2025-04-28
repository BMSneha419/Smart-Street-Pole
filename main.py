import sys
import torch
import ctypes,os
from track import detect
from tkinter import *
from time import sleep
from tkinter.tix import *
from PIL import ImageTk, Image
import tkinter.messagebox as tkMessageBox
import tkinter.filedialog as filedialog
from deep_sort_pytorch.deep_sort import opt
from deep_sort_pytorch.utils.parser import get_config

def HomePage():
	global about, imgmode, vdomode
	try:
		about.destroy()
	except:
		pass
	try:
		imgmode.destroy()
	except:
		pass
	try:
		vdomode.destroy()
	except:
		pass

	window = Tk()
	img = Image.open("Images\\HomePage.png")
	img = ImageTk.PhotoImage(img)
	panel = Label(window, image=img)
	panel.pack(side="top", fill="both", expand="yes")

	user32 = ctypes.windll.user32
	user32.SetProcessDPIAware()
	[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
	lt = [w, h]
	a = str(lt[0]//2-446)
	b= str(lt[1]//2-383)

	window.title("HOME - Smart Street Pole")
	window.geometry("904x533+"+a+"+"+b)
	window.resizable(0,0)



	def aboutus():
		global about
		try:
			window.destroy()
		except:
			pass
		

		about = Tk()
		img = Image.open("Images\\AboutUs.png")
		img = ImageTk.PhotoImage(img)
		panel = Label(about, image=img)
		panel.pack(side="top", fill="both", expand="yes")

		user32 = ctypes.windll.user32
		user32.SetProcessDPIAware()
		[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
		lt = [w, h]
		a = str(lt[0]//2-446)
		b= str(lt[1]//2-383)

		about.title("ABOUT US - Smart Street Pole")
		about.geometry("904x533+"+a+"+"+b)
		about.resizable(0,0)

		homebtn = Button(about,text = "HomePage",font = ("Agency FB",16,"bold"),relief = FLAT, bd = 0, borderwidth='0',bg="#1A1D24",fg="#D5DAE0",activebackground = "#1A1D24",activeforeground = "#D5DAE0",command=HomePage)	
		homebtn.place(x=674, y = 20)
			
		exitbtn = Button(about,text = "Exit",font = ("Agency FB",16,"bold"),relief = FLAT, bd = 0, borderwidth='0',bg="#191C23",fg="#D5DAE0",activebackground = "#191C23",activeforeground = "#D5DAE0",command=exit)
		exitbtn.place(x=782,y = 20)

		about.mainloop()

	# EXIT . . . 
	def exit():
		global about
		result = tkMessageBox.askquestion("Smart Street Pole", "Are you sure you want to exit?", icon= "warning")
		if result == 'yes':
			sys.exit()


	def browsevideo():
		cfg = get_config()
		cfg.merge_from_file('deep_sort_pytorch/configs/deep_sort.yaml')
		filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file", filetypes=( ("Video Files",(".mp4",".avi",".mkv")),("All Files", "*.*")))
		opt_obj = opt.Opt(
			source = filename,
			save_vid = cfg.DETECTION.SAVE_VID,
			save_txt= cfg.DETECTION.SAVE_TXT,
			upper= cfg.DETECTION.UPPER,
			lower=cfg.DETECTION.LOWER,
			motorcycle=cfg.DETECTION.MOTORCYCLE,
			car=cfg.DETECTION.CAR,
			truck=cfg.DETECTION.TRUCK,
			bus=cfg.DETECTION.BUS,
			distance=cfg.DETECTION.DISTANCE,
			track = cfg.DETECTION.TRACK,
			bbox = cfg.DETECTION.BBOX
		)
		try:
			with torch.no_grad():
				detect(opt_obj)
		except:
			pass

	
	''' MENU BAR '''             
	aboutusbtn = Button(window,text = "About Us",font = ("Agency FB",16,"bold"),relief = FLAT, bd = 0, borderwidth='0',bg="#000000",fg="#948B8B",activebackground = "#000000",activeforeground = "#948B8B",command=aboutus)	
	aboutusbtn.place(x=674, y = 10)

	exitbtn = Button(window,text = "Exit",font = ("Agency FB",16,"bold"),relief = FLAT, bd = 0, borderwidth='0',bg="#000000",fg="#948B8B",activebackground = "#000000",activeforeground = "#948B8B",command=exit)
	exitbtn.place(x=782,y = 10)

	videobtn = Button(window,text = "UPLOAD VIDEO",font = ("Arial Narrow",16,"bold"),width = 15,relief = FLAT, bd = 1, borderwidth='1',bg="#59070F",fg="#807676",activebackground = "#59070F",activeforeground = "#807676",command=browsevideo)
	videobtn.place(x=75,y = 342)
	

	window.mainloop()

#HomePage()
def LoadingScreen():
	root = Tk()
	root.config(bg="white")
	root.title("Loading - Smart Street Pole")

	img = Image.open(r"Images\\Loading.png")
	img = ImageTk.PhotoImage(img)
	panel = Label(root, image=img)
	panel.pack(side="top", fill="both", expand="yes")

	user32 = ctypes.windll.user32
	user32.SetProcessDPIAware()
	[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
	lt = [w, h]
	a = str(lt[0]//2-446)
	b= str(lt[1]//2-383)

	root.geometry("902x553+"+a+"+"+b)
	root.resizable(0,0)

	for i in range(27):
		Label(root, bg="#574D72",width=2,height=1).place(x=(i+4)*25,y=520) 

	def play_animation(): 
		for j in range(27):
			Label(root, bg= "#9477CD",width=2,height=1).place(x=(j+4)*25,y=520) 
			sleep(0.17)
			root.update_idletasks()
		else:
			root.destroy()
			HomePage()

	root.update()
	play_animation()
	root.mainloop()
	
LoadingScreen()
