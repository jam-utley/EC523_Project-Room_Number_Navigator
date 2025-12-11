import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import json
import os
import glob
import re

class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("VLM Annotation Tool")
        self.root.geometry("1200x800")

        # Data state
        self.image_dir = ""
        self.image_files = []
        self.current_image_index = 0
        self.metadata = []  # List of dicts: {"image": "filename", "annotation": "string"}
        self.current_boxes = [] # List of [x1, y1, x2, y2, label]
        self.scale_factor = 1.0
        self.image_path = None

        # UI Components
        self._setup_ui()
        
        # Canvas events
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.root.bind("<Delete>", self.delete_selected_box)

        self.selected_box_index = -1

    def _setup_ui(self):
        # Top Control Panel
        control_frame = tk.Frame(self.root, bd=2, relief=tk.RAISED)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(control_frame, text="Load Directory", command=self.load_directory).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Save Metadata", command=self.save_metadata, bg="#ddffdd").pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text=" | ").pack(side=tk.LEFT)

        tk.Button(control_frame, text="Clean Dataset", command=self.clean_dataset, bg="#ffcccc").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Delete Image", command=self.delete_current_image, bg="#ffaaaa").pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text=" | ").pack(side=tk.LEFT)
        
        tk.Button(control_frame, text="<< Prev", command=self.prev_image).pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="Next >>", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        self.lbl_status = tk.Label(control_frame, text="No directory loaded")
        self.lbl_status.pack(side=tk.LEFT, padx=20)

        # Main Layout
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Left: Canvas
        self.canvas_frame = tk.Frame(main_pane, bg="gray")
        main_pane.add(self.canvas_frame)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right: Listbox for detections
        right_frame = tk.Frame(main_pane, width=250)
        main_pane.add(right_frame)
        
        tk.Label(right_frame, text="Detections (Select to Delete)").pack(side=tk.TOP, pady=5)
        self.box_list = tk.Listbox(right_frame)
        self.box_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.box_list.bind("<<ListboxSelect>>", self.on_listbox_select)
        
        tk.Button(right_frame, text="Delete Selected", command=self.delete_selected_box, bg="#ffdddd").pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    def load_directory(self):
        directory = filedialog.askdirectory()
        if not directory:
            return

        self.image_dir = directory
        # Find images
        exts = ["*.png", "*.jpg", "*.jpeg"]
        self.image_files = []
        for ext in exts:
            self.image_files.extend(glob.glob(os.path.join(directory, ext)))
        self.image_files.sort()

        if not self.image_files:
            messagebox.showwarning("No Images", "No images found in selected directory.")
            return

        # Load metadata if exists
        meta_path = os.path.join(directory, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load metadata.json: {e}")
                self.metadata = []
        else:
            self.metadata = []
            # Initialize metadata for all images
            for img_path in self.image_files:
                fname = os.path.basename(img_path)
                self.metadata.append({"image": fname, "annotation": "[DETECTIONS: ]"})

        # Sync metadata with current files (add new files if any)
        existing_fnames = {m["image"] for m in self.metadata}
        for img_path in self.image_files:
            fname = os.path.basename(img_path)
            if fname not in existing_fnames:
                self.metadata.append({"image": fname, "annotation": "[DETECTIONS: ]"})
        
        self.current_image_index = 0
        self.load_image()

    def load_image(self):
        if not self.image_files:
            return

        img_path = self.image_files[self.current_image_index]
        self.image_path = img_path
        fname = os.path.basename(img_path)
        
        # Update status
        self.lbl_status.config(text=f"Image {self.current_image_index + 1}/{len(self.image_files)}: {fname}")

        # Load Image
        pil_image = Image.open(img_path)
        
        # Resize for display if needed
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: cw, ch = 800, 600 # Default if not yet rendered
        
        iw, ih = pil_image.size
        self.scale_factor = min(cw/iw, ch/ih) # Scale to fit (up or down)
        
        new_w = int(iw * self.scale_factor)
        new_h = int(ih * self.scale_factor)
        
        self.tk_image = ImageTk.PhotoImage(pil_image.resize((new_w, new_h)))
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Load annotations
        self.current_boxes = []
        meta_entry = next((m for m in self.metadata if m["image"] == fname), None)
        if meta_entry:
            self.current_boxes = self.parse_annotation(meta_entry["annotation"])
        
        self.redraw_boxes()
        self.update_listbox()

    def parse_annotation(self, annotation_str):
        # Format: [DETECTIONS: [BOX: x1, y1, x2, y2, "label"], ...]
        # Regex to find inner BOX parts
        # Example inner: [BOX: 271, 100, 299, 128, "nine"]
        pattern = r'\[BOX:\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*"([^"]+)"\]'
        matches = re.findall(pattern, annotation_str)
        boxes = []
        for m in matches:
            x1, y1, x2, y2, label = m
            boxes.append([int(x1), int(y1), int(x2), int(y2), label])
        return boxes

    def generate_annotation_string(self, boxes):
        if not boxes:
            return "[DETECTIONS: ]"
        
        items = []
        for b in boxes:
            x1, y1, x2, y2, label = b
            items.append(f'[BOX: {x1}, {y1}, {x2}, {y2}, "{label}"]')
        return "[DETECTIONS: " + ", ".join(items) + "]"

    def save_current_annotation(self):
        if not self.image_path: return
        fname = os.path.basename(self.image_path)
        
        # Update metadata entry
        for m in self.metadata:
            if m["image"] == fname:
                m["annotation"] = self.generate_annotation_string(self.current_boxes)
                break

    def save_metadata(self):
        self.save_current_annotation() # Save current in-memory first
        if not self.image_dir: return
        
        path = os.path.join(self.image_dir, "metadata.json")
        try:
            with open(path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            messagebox.showinfo("Saved", f"Metadata saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def clean_dataset(self):
        if not self.image_dir: return
        if not messagebox.askyesno("Confirm Clean", "This will DELETE all images that have no annotations. Are you sure?"):
            return

        self.save_current_annotation()
        
        images_to_delete = []
        new_metadata = []
        
        for m in self.metadata:
            # Check if annotation is empty or just has empty detections
            if m["annotation"] == "[DETECTIONS: ]" or not self.parse_annotation(m["annotation"]):
                images_to_delete.append(m["image"])
            else:
                new_metadata.append(m)
        
        if not images_to_delete:
            messagebox.showinfo("Clean", "No unannotated images found.")
            return

        count = 0
        for fname in images_to_delete:
            path = os.path.join(self.image_dir, fname)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    count += 1
                except Exception as e:
                    print(f"Failed to delete {path}: {e}")
        
        self.metadata = new_metadata
        self.save_metadata()
        
        # Refresh file list
        self.image_files = [f for f in self.image_files if os.path.basename(f) not in images_to_delete]
        self.current_image_index = 0
        self.load_image()
        
        messagebox.showinfo("Clean Complete", f"Deleted {count} unannotated images.")

    def delete_current_image(self):
        if not self.image_path: return
        if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {os.path.basename(self.image_path)}?"):
            return

        fname = os.path.basename(self.image_path)
        
        # Remove file
        if os.path.exists(self.image_path):
            try:
                os.remove(self.image_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete file: {e}")
                return

        # Remove from metadata
        self.metadata = [m for m in self.metadata if m["image"] != fname]
        self.save_metadata()

        # Remove from list and navigate
        del self.image_files[self.current_image_index]
        
        if not self.image_files:
            self.canvas.delete("all")
            self.lbl_status.config(text="No images left.")
            self.image_path = None
            self.current_boxes = []
            self.update_listbox()
            return

        if self.current_image_index >= len(self.image_files):
            self.current_image_index = len(self.image_files) - 1
        
        self.load_image()

    def next_image(self):
        self.save_current_annotation()
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image()

    def prev_image(self):
        self.save_current_annotation()
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    # --- Canvas Interaction ---

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2
        )

    def on_mouse_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x, end_y = (event.x, event.y)
        
        # Normalize coords (handle dragging up/left)
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        # Ignore tiny boxes
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            self.canvas.delete(self.current_rect)
            return

        # Ask for label
        label = simpledialog.askstring("Input", "Enter Label:")
        if label:
            # Scale back to original image coordinates
            orig_x1 = int(x1 / self.scale_factor)
            orig_y1 = int(y1 / self.scale_factor)
            orig_x2 = int(x2 / self.scale_factor)
            orig_y2 = int(y2 / self.scale_factor)
            
            self.current_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2, label])
            self.redraw_boxes()
            self.update_listbox()
        else:
            self.canvas.delete(self.current_rect)

    def redraw_boxes(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        for i, box in enumerate(self.current_boxes):
            x1, y1, x2, y2, label = box
            # Scale to display
            sx1 = x1 * self.scale_factor
            sy1 = y1 * self.scale_factor
            sx2 = x2 * self.scale_factor
            sy2 = y2 * self.scale_factor
            
            color = "red" if i == self.selected_box_index else "blue"
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2)
            self.canvas.create_text(sx1, sy1, text=label, anchor=tk.SW, fill=color, font=("Arial", 10, "bold"))

    def update_listbox(self):
        self.box_list.delete(0, tk.END)
        for box in self.current_boxes:
            self.box_list.insert(tk.END, f"{box[4]} : {box[0]},{box[1]},{box[2]},{box[3]}")

    def on_listbox_select(self, event):
        selection = self.box_list.curselection()
        if selection:
            self.selected_box_index = selection[0]
            self.redraw_boxes()

    def delete_selected_box(self, event=None):
        if self.selected_box_index >= 0 and self.selected_box_index < len(self.current_boxes):
            del self.current_boxes[self.selected_box_index]
            self.selected_box_index = -1
            self.redraw_boxes()
            self.update_listbox()

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()
