import tkinter as tk
from PIL import Image, ImageDraw
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multiple Digit Recognizer")
        
        if not os.path.exists("digit_model.h5"):
            print("Please run train.py first to generate the model!")
            self.root.destroy()
            return

        self.model = load_model("digit_model.h5")

        self.canvas_width = 600
        self.canvas_height = 250
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black', cursor="cross")
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()
        tk.Button(btn_frame, text="Predict", command=self.predict_digits, font=("Arial", 14), bg="green", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Clear", command=self.clear_canvas, font=("Arial", 14), bg="red", fg="white").pack(side=tk.LEFT, padx=10)

        self.result_label = tk.Label(self.root, text="Result: ", font=("Arial", 24, "bold"))
        self.result_label.pack(pady=20)

    def paint(self, event):
        r = 8 
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Result: ")

    def predict_digits(self):
        img_array = np.array(self.image)

        contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return

        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        sorted_contours = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][0])]

        final_prediction = ""

        for c in sorted_contours:
            x, y, w, h = cv2.boundingRect(c)
            
            if w < 10 or h < 10:
                continue

            roi = img_array[y:y+h, x:x+w]

            max_dim = max(w, h) + 20
            square = np.zeros((max_dim, max_dim), dtype=np.uint8)
            
            start_x = (max_dim - w) // 2
            start_y = (max_dim - h) // 2
            square[start_y:start_y+h, start_x:start_x+w] = roi

            resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

            normalized = resized / 255.0
            reshaped = normalized.reshape(1, 28, 28, 1)

            prediction = self.model.predict(reshaped)
            digit = np.argmax(prediction)
            final_prediction += str(digit)

        self.result_label.config(text=f"Result: {final_prediction}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()