"""
- SegContour

This program automatically generates contours around mice in grayscale 
videos with a bright background. The contours are created by detecting 
edges around the mice.

Before running this program, ensure that the input videos have been 
color-segmented properly.

To use this program:
1. Specify the input video files.
2. Specify the output directory for the processed videos.
3. Click the 'Process' button to apply contours to the videos.

Note: The program's performance may vary depending on recording 
conditions, environment, and the quality of the color segmentation.

"""

import os, cv2, numpy as np, tkinter as tk
from scipy.signal import find_peaks
from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor

def LoadVideo(video_path: str) -> tuple[cv2.VideoCapture, int, int, int, int]:
    """
    The LoadVideo function loads the input video and retrieves basic 
    properties, such as frame dimensions, frame rate, and total frame count.

    Args:
        video_path (str): 
            The path to the source video where the contours will be added.

    Returns:
        cap (cv2.VideoCapture): 
            A VideoCapture object to read frames from the video file.
        
        frame_width (int): 
            The width of each video frame in pixels.

        frame_height (int): 
            The height of each video frame in pixels.

        fps (int): 
            The frame rate of the video (frames per second).

        frame_count (int): 
            The total number of frames in the video.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Retrieve the width of the video frames in pixels
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Retrieve the height of the video frames in pixels
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Retrieve the frame rate (frames per second) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Retrieve the total number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Return the VideoCapture object and the extracted video properties
    return cap, frame_width, frame_height, fps, frame_count

def CalculateThreshold(video_path: str) -> tuple[int, int, int]:
    """
    The CalculateThreshold function Calculate the thresholds for each 
    color channel (Blue, Green, Red) based on the average histograms of 
    the video frames.

    Args:
        video_path (str):
            The path to the source video.

    Returns:
        threshold_b (int):
            Calculated threshold value for the Blue channel.
            
        threshold_g (int):
            Calculated threshold value for the Green channel.

        threshold_r (int):
            Calculated threshold value for the Red channel.
    """

    # Load the video and get the total number of frames
    cap, _, _, _, frame_count = LoadVideo(video_path)

    # Initialize histograms for Blue, Green, and Red channels with 256 bins
    hist_b_total = np.zeros((256, 1))
    hist_g_total = np.zeros((256, 1))
    hist_r_total = np.zeros((256, 1))

    # Counter to track the number of processed frames
    processed_frames = 0

    # Loop through all frames in the video
    for _ in range(frame_count):
        # Read the next frame from the video
        res, frame = cap.read()
        if not res:
            break

        # Split the frame into Blue, Green, and Red channels
        b, g, r = cv2.split(frame)

        # Calculate histograms for each channel
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

        # Accumulate histograms across frames
        hist_b_total += hist_b
        hist_g_total += hist_g
        hist_r_total += hist_r

        # Increment the number of processed frames
        processed_frames += 1

    # Calculate average histograms for each channel
    hist_b_avg = hist_b_total / processed_frames
    hist_g_avg = hist_g_total / processed_frames
    hist_r_avg = hist_r_total / processed_frames

    # Find peaks in the histograms for each channel
    peaks_b, _ = find_peaks(-hist_b_avg.ravel(), distance=5, prominence=10)
    peaks_g, _ = find_peaks(-hist_g_avg.ravel(), distance=5, prominence=10)
    peaks_r, _ = find_peaks(-hist_r_avg.ravel(), distance=5, prominence=10)

    # Select the 5th peak as the threshold for each channel
    threshold_b = peaks_b[4]
    threshold_g = peaks_g[4]
    threshold_r = peaks_r[4]

    # Release the video capture object
    cap.release()

    # Return the calculated thresholds for each channel
    return threshold_b, threshold_g, threshold_r

def ContourExtraction(image: np.ndarray, channel: np.ndarray) -> np.ndarray:
    """
    The ContourExtraction function extracts contours from the given image 
    by detecting edges in the specified channel.

    Args:
        image (np.ndarray):
            The original input image (in BGR format).
        
        channel (np.ndarray):
            The single-channel grayscale image where edge detection will 
            be applied.

    Returns:
        final_image (np.ndarray):
            The resulting image with contours overlaid on the original 
            image.
    """

    # Apply Canny edge detection on the specified channel
    edges = cv2.Canny(channel, 150, 200)

    # Convert the edges to a 3-channel (BGR) image for overlaying
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Create an overlay by replacing edge pixels with white (255, 255, 255)
    overlay = np.where(edges_colored == 255, (255, 255, 255), image)

    # Ensure the overlay is in the correct data type for OpenCV operations
    overlay = overlay.astype(np.uint8)

    # Blend the original image and the overlay to create the final image
    final_image = cv2.addWeighted(image, 0, overlay, 1, 0)

    # Return the final image with contours
    return final_image

def MakeContouredVideo(video_path: str, output_video_path: str, threshold_b: int, threshold_g: int, threshold_r: int, progress_callback: function) -> None:
    """
    The MakeContouredVideo function processes a video to generate contours 
    based on thresholds and save the output.

    Args:
        video_path (str):
            The path to the input video.
        
        output_video_path (str):
            The path where the processed video with contours will be saved.

        threshold_b (int):
            Threshold value for the Blue channel.

        threshold_g (int):
            Threshold value for the Green channel.

        threshold_r (int):
            Threshold value for the Red channel.

        progress_callback (function):
            A callback function to update progress, accepting current 
            frame and total frames as arguments.
    
    Returns:
        None
    """

    # Load the video and retrieve its properties
    cap, frame_width, frame_height, fps, frame_count = LoadVideo(video_path)

    # Define the codec and create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame in the video
    for i in range(frame_count):
        # Read the next frame from the video
        res, frame = cap.read()
        if not res:
            break

        # Split the frame into Blue, Green, and Red channels
        b, g, r = cv2.split(frame)

        # Apply thresholding to each channel
        b[b >= threshold_b] = 255
        g[g >= threshold_g] = 255
        r[r >= threshold_r] = 255

        # Merge the thresholded channels back into a single image
        merged_image = cv2.merge((b, g, r))

        # Convert the merged image to grayscale for contour extraction
        gray = cv2.cvtColor(merged_image, cv2.COLOR_BGR2GRAY)

        # Extract contours and generate the final frame
        final_frame = ContourExtraction(frame, gray)

        # Write the processed frame to the output video
        out.write(final_frame)

        # Update progress using the callback function
        progress_callback(i + 1, frame_count)

    # Release the video capture and writer resources
    cap.release()
    out.release()

# The GUI class definition
class SegContourGUI:
    """
    The SegContourGUI class provides a GUI for the SegContour application.
    Users can select input video files and specify an output directory to 
    process videos with contouring applied based on calculated thresholds.

    This class manages the user interface interactions, including selecting 
    files and folders, displaying progress, and initiating video processing.

    Args:
        root: A Tkinter root window object for the application GUI.

    Methods:
        BrowseFiles():
            Opens a file dialog to select multiple input video files and 
            updates the listbox with the selected file paths.

        BrowseFolder():
            Opens a file dialog to select an output directory and 
            updates the listbox with the selected directory path.

        UpdateProgress(video_name, current_frame, total_frames):
            Updates the progress bar and label to reflect the current frame 
            during video processing.

        ShowProgressWidgets():
            Displays the progress bar and label during video processing.

        HideProgressWidgets():
            Hides the progress bar and label when processing is complete or 
            canceled.

        StartProcessing():
            Validates input and output paths, then starts the video processing 
            operation in a separate thread to keep the GUI responsive.
    """

    def __init__(self, root):
        """
        Initializes the SegContourGUI class by setting up the GUI components, 
        configuring the layout, and initializing input and output paths.
        
        Args:
            root: The Tkinter root window object for the GUI.
        
        Attributes:
            root (tk.Tk):
                The root window object for the GUI.

            input_listbox (tk.Listbox):
                Listbox widget for displaying selected input video files.

            output_listbox (tk.Listbox):
                Listbox widget for displaying the selected output directory.
                
            progress_label (tk.Label):
                Label widget for displaying the current progress of video 
                processing.

            progress_bar (ttk.Progressbar):
                Progressbar widget for showing the percentage of video 
                processing completed.

            input_scroll_y (ttk.Scrollbar):
                Vertical scrollbar for the input video listbox.

            input_scroll_x (ttk.Scrollbar):
                Horizontal scrollbar for the input video listbox.

            output_scroll_x (ttk.Scrollbar):
                Horizontal scrollbar for the output directory listbox.
        """

        self.root = root
        self.root.title("SegContour GUI")

        # Set initial size of the GUI
        self.root.geometry("600x150")

        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)  # Input Video Paths row
        self.root.grid_rowconfigure(1, weight=1)  # Scrollbar for Input Videos
        self.root.grid_rowconfigure(2, weight=0)  # Output Directory row (fixed height)
        self.root.grid_rowconfigure(3, weight=0)  # Scrollbar for Output Directory
        self.root.grid_rowconfigure(4, weight=0)  # Progress Label row
        self.root.grid_columnconfigure(0, weight=0)  # Labels column (fixed width)
        self.root.grid_columnconfigure(1, weight=1)  # Main columns (Listboxes)
        self.root.grid_columnconfigure(2, weight=0)  # Buttons column (fixed width)

        # Input Videos Section
        tk.Label(root, text="Input Videos:", font=("Arial", 10)).grid(row=0, column=0, sticky='e', padx=5, pady=5)

        # Frame for input video listbox and scrollbars
        input_listbox_frame = tk.Frame(root)
        input_listbox_frame.grid(row=0, column=1, padx=10, pady=1, sticky="nsew")

        # Listbox for displaying input video file paths
        self.input_listbox = tk.Listbox(input_listbox_frame, height=1, font=("Arial", 10))
        self.input_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Vertical scrollbar for input listbox
        self.input_scroll_y = ttk.Scrollbar(input_listbox_frame, orient=tk.VERTICAL, command=self.input_listbox.yview)
        self.input_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar for input listbox
        self.input_scroll_x = ttk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.input_listbox.xview)
        self.input_scroll_x.grid(row=1, column=1, padx=10, pady=1, sticky="ew")

        # Attach scrollbars to listbox
        self.input_listbox.config(yscrollcommand=self.input_scroll_y.set, xscrollcommand=self.input_scroll_x.set)

        # Initially hide scrollbars
        self.input_scroll_y.pack_forget()
        self.input_scroll_x.grid_remove()

        # Browse button to select input videos
        tk.Button(root, text="Browse", command=self.BrowseFiles).grid(row=0, column=2, padx=5, pady=5)

        # Output Directory Section
        tk.Label(root, text="Output Directory:", font=("Arial", 10)).grid(row=2, column=0, sticky='e', padx=5, pady=5)

        # Frame for output directory listbox
        output_listbox_frame = tk.Frame(root)
        output_listbox_frame.grid(row=2, column=1, padx=10, pady=1, sticky="ew")

        # Listbox for displaying output directory path
        self.output_listbox = tk.Listbox(output_listbox_frame, height=1, font=("Arial", 10))
        self.output_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Horizontal scrollbar for output listbox
        self.output_scroll_x = ttk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.output_listbox.xview)
        self.output_scroll_x.grid(row=3, column=1, padx=10, pady=1, sticky="ew")

        # Attach scrollbar to output listbox
        self.output_listbox.config(xscrollcommand=self.output_scroll_x.set)

        # Initially hide horizontal scrollbar
        self.output_scroll_x.grid_remove()

        # Browse button to select output folder
        tk.Button(root, text="Browse", command=self.BrowseFolder).grid(row=2, column=2, padx=5, pady=5)

        # Progress Bar Section
        self.progress_label = tk.Label(root, text="Progress: N/A", font=("Arial", 10))
        self.progress_label.grid(row=4, column=1, columnspan=1, sticky=tk.W, padx=5, pady=5)
        self.progress_label.grid_remove()

        self.progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=400)
        self.progress_bar.grid(row=5, column=1, padx=10, pady=10)
        self.progress_bar.grid_remove()

        # Action Buttons
        tk.Button(root, text="Process", command=self.StartProcessing).grid(row=5, column=0, pady=20, padx=(10, 0))
        tk.Button(root, text="Close", command=root.quit).grid(row=5, column=2, pady=20, padx=(0, 10))

    def BrowseFiles(self):
        """
        Opens a file dialog to allow the user to select multiple input 
        video files. Updates the listbox with the selected file paths 
        and adjusts the visibility of scrollbars based on the number and 
        length of the file paths.

        Returns:
            None
        """

        # Open file dialog to select video files
        file_paths = filedialog.askopenfilenames(title="Select Video Files", filetypes=[("Video Files", "*.mp4;*.avi")])
        if file_paths:
            self.input_listbox.delete(0, tk.END)
            for path in file_paths:
                self.input_listbox.insert(tk.END, path)

            # Show or hide vertical scrollbar based on the number of videos
            listbox_height = int(self.input_listbox.cget("height"))
            if len(file_paths) > listbox_height:
                self.input_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                self.input_scroll_y.pack_forget()

            # Show or hide horizontal scrollbar based on file path length
            if any(len(path) > 50 for path in file_paths):  # Assuming 50 chars is the threshold
                self.input_scroll_x.grid()
            else:
                self.input_scroll_x.grid_remove()

    def BrowseFolder(self):
        """
        Opens a file dialog to allow the user to select an output folder 
        and updates the listbox with the selected directory path. Adjusts 
        the visibility of the scrollbar based on the length of the folder 
        path.

        Returns:
            None
        """

        # Open file dialog to select output folder
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_listbox.delete(0, tk.END)
            self.output_listbox.insert(tk.END, folder_path)

            # Show or hide horizontal scrollbar if the path is too long
            if len(folder_path) > 50:
                self.output_scroll_x.grid()
            else:
                self.output_scroll_x.grid_remove()

    def UpdateProgress(self, video_name: str, current_frame: int, total_frames: int):
        """
        Updates the progress bar and label to reflect the current frame 
        during video processing.

        Args:
            video_name (str):
                The name of the currently processed video.

            current_frame (int):
                The current frame being processed.
                
            total_frames (int):
                The total number of frames in the video.

        Returns:
            None
        """

        progress_percentage = (current_frame / total_frames) * 100
        self.progress_bar["value"] = progress_percentage
        self.progress_label.config(text=f"Processing {video_name}: Frame {current_frame}/{total_frames}")
        self.root.update_idletasks()

    def ShowProgressWidgets(self):
        """
        Displays the progress bar and label during video processing.

        Returns:
            None
        """

        self.progress_label.grid()
        self.progress_bar.grid()

    def HideProgressWidgets(self):
        """
        Hides the progress bar and label when processing is complete or 
        canceled.

        Returns:
            None
        """

        self.progress_label.grid_remove()
        self.progress_bar.grid_remove()

    def StartProcessing(self):
        """
        Validates whether input videos and an output directory are selected, 
        and starts the video processing operation in a separate thread to 
        keep the GUI responsive. If the input or output is missing, an error 
        message is displayed.

        Returns:
            None
        """
        # Retrieve the list of input video paths from the listbox
        video_paths = self.input_listbox.get(0, tk.END)

        # Retrieve the output directory path from the listbox
        output_video_folder = self.output_listbox.get(0, tk.END)[0] if self.output_listbox.size() > 0 else ""

        # Check if input videos are selected
        if not video_paths:
            messagebox.showerror("Error", "Please select input videos.")
            return

        # Check if output directory is selected
        if not output_video_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        # Show the progress bar and label while processing
        self.ShowProgressWidgets()

        # Define the function to process the videos
        def ProcessVideos():
            for video_path in video_paths:
                video_name = os.path.basename(video_path)

                # Define the output path for the processed video
                output_video_path = os.path.join(output_video_folder, f"processed_{video_name}")

                # Calculate the threshold values for each color channel (B, G, R)
                threshold_b, threshold_g, threshold_r = CalculateThreshold(video_path)

                # Process the video and apply contours
                MakeContouredVideo(
                    video_path,
                    output_video_path,
                    threshold_b,
                    threshold_g,
                    threshold_r,
                    # Update progress during processing
                    lambda current_frame, total_frames: self.UpdateProgress(video_name, current_frame, total_frames)
                )

            # Hide the progress bar and label after processing
            self.HideProgressWidgets()

            # Show a success message when all videos have been processed
            messagebox.showinfo("Info", "All videos have been processed successfully.")

        # Run the video processing operation in a separate thread to keep the GUI responsive
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(ProcessVideos)

if __name__ == "__main__":
    root = tk.Tk()
    app = SegContourGUI(root)
    root.mainloop()