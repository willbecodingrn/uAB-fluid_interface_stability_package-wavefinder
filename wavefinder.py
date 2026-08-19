'''
This module will import a video, analyze each frame to calculate the interface stability of that frame, and return a time series for interface stability.
1. import libraries
2. import video
3. define tracker density, start and end frames, time steps, and viewport times
4. assign dots to interface and cenrte, and track dots across time
5. compute tracker statistics (statistical method)
    or
6. compute fingering geometry (geometric method, probably more expensive but accurate)
'''
#import as needed
import cv2
import time
import numpy as np
import sys

'''
path = "jack_data/Exp videos/"

currpath = path + "fluid 1/MVI_0449.MP4"
'''

#region progress bar
def prog_bar(curr, total, length=30, text=None):
    percent = float(curr)/total
    arrow = 'x' * int(round(percent*length))
    spaces = '-' * (length-len(arrow))
    if text: text += ': '
    sys.stdout.write(f'\r{text}[{arrow}{spaces}] {percent*100:.2f}%')
    sys.stdout.flush()

def clear_bar():
    sys.stdout.write('\r\033[K')
    sys.stdout.flush()
#endregion

def polarize(coord, centre=(0,0)):
    r = np.sqrt((coord[0]-centre[0])**2 + (coord[1]-centre[1])**2)
    theta = np.atan2((coord[0] - centre[0]), (coord[1] - centre[1]))
    return float(r), float(theta)

def resample(contour, pointCount=200):
    points = contour.reshape(-1,2).astype(np.float64)

    if len(points) <2: return None 

    points_closed = np.vstack([points, points[0]])
    semgent_lengths = np.linalg.norm(np.diff(points_closed, axis=0), axis=1)
    cumulative_length = np.concatenate(([0], np.cumsum(semgent_lengths)))
    total_length = cumulative_length[-1]

    if total_length == 0: return None

    sample_pos = np.linspace(0, total_length, pointCount, endpoint=False)

    x = np.interp(sample_pos, cumulative_length, points_closed[:, 0])
    y = np.interp(sample_pos, cumulative_length, points_closed[:, 1])
    return np.column_stack((x,y))

def translate(frame, prevCentroid=None, pointCount=200, min_area=50, roi_fraction=0.3, max_distance=None, console=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f_height, f_width = frame.shape[:2]

    #increasing local constrast to try to fix broken contours
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    #region geometry
    edges = cv2.Canny(blurred, threshold1=20, threshold2=90) #adjust these
    #kernel = np.ones((3,3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) #resewing, was 5,5; might need to be odd
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    #debbugging windows
    cv2.imshow("Canny", edges)
    #cv2.imshow("gray", gray)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if len(contours) == 0:
        print('no contours found')
        return None
    #endregion
    #main_contour = max(contours, key=cv2.contourArea)

    #region find contour
    candidates = []

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        can_radii = np.sqrt(np.sum((c-[cx,cy])**2, axis=-1))
        if 2*max(can_radii) > (f_height*0.95): continue
        else:
            candidates.append({"contour": c, "centroid": (cx, cy), "area":cv2.contourArea(c)})

    if len(candidates) ==0:
        if console: print("no valid contour centroids found")
        return None

    #region frame logic
    if prevCentroid == None:
        #region init ROI
        centrex = f_width / 2
        centrey = f_height /2
        roi_width = f_width * roi_fraction
        roi_height = f_height * roi_fraction

        xmin = centrex - roi_width/2
        xmax = centrex + roi_width/2
        ymin = centrey - roi_height/2
        ymax = centrey + roi_height/2
        #endregion
        roi_candidates = []

        for c in candidates:
            cx, cy = c["centroid"]
            if(xmin <= cx <= xmax and ymin <= cy <= ymax): roi_candidates.append(c)
        if len(roi_candidates) == 0:
            if console: print('no contour centroid found within ROI')
            return None

        '''
        implement nuanced selection criteria:
            > combines considerations in area, average radii, distance to previous interation
        '''

        selected = max(roi_candidates, key=lambda candidate: candidate["area"])

    else:
        prevx, prevy = prevCentroid
        def from_prev(candidate):
            cx, cy = candidate["centroid"]
            return np.sqrt((cx - prevx)**2 + (cy - prevy)**2)

        selected = min(candidates, key=from_prev)
        d = from_prev(selected)
        if max_distance and (d > max_distance):
            if console: print(f'contour step exceeds allowed distance')
            if console: print(f'nearest distance = {d:.2f} pixels')
            return None

    main_contour = selected["contour"]
    cx, cy = selected["centroid"]
    #endregion

    #region data
    points = resample(main_contour, pointCount)
    if points is None: return None

    #endregion
    annotated = frame.copy()
    cv2.drawContours(annotated, [main_contour], -1, (0, 255, 0), 2)

    if not np.isnan(cx):
        cv2.circle(annotated, (int(cx), int(cy)), 5, (0,0,255), -1)

    return points, annotated

def stability(coords):
    coords = coords.T
    x_list = coords[0]
    y_list = coords[1]
    centre = (np.average(x_list), np.average(y_list))
    radii = np.sqrt((x_list - centre[0])**2 + (y_list - centre[1])**2)
    return np.std(radii)

def retina(path, livePlay=True, save=False, singular=True, nc_points=200, showText=False, display=True):
    #region init video
    cap = cv2.VideoCapture(path)
    if not display: livePlay = False

    if not cap.isOpened():
        print("Error: could not open video")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count/fps

    c_history = np.full((frame_count, nc_points, 2), np.nan)

    print(f"FPS: {fps:.3f}\t\tframe count: {frame_count}")
    print(f"Resolution: {width} x {height}\t\t\tduration: {frame_count / fps:.2f}")
    #endregion

    currFrame = 0
    timechart = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    #region sweep
    while True:
        ret, frame = cap.read()
        if not ret: break

        time_seconds = currFrame / fps
        text = f"Frame = {currFrame}, " + f"Time = {time_seconds: .4f}"
        if showText: print(text)
        text += f"\nduration: {duration:.2f} \nq: close"

        raw = frame.copy()
        result = translate(raw, pointCount=nc_points, console=showText)
        if result is not None:
            c_points, shown_frame = result
            c_history[currFrame] = c_points
            curr_stable = stability(c_points)
        else: 
            shown_frame = raw.copy()
            curr_stable = np.nan

        if showText: print(f'stability score: {curr_stable:.2f}\n')
        text += f'\ninstability: {curr_stable:.2f}'
        timechart.append([currFrame,curr_stable])

        if display:
            cv2.putText(img=shown_frame, text=text, org=(20, 40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            cv2.imshow(path, shown_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"): break
        currFrame += 1
        if livePlay: time.sleep(1/fps)
        prog_bar(currFrame, frame_count, text='\tscanning current video')
    #endregion

    timechart = np.array(timechart)
    cap.release()
    clear_bar()
    if display: cv2.destroyAllWindows()
    
    if save and singular: 
        np.savetxt(path[-12:-4]+'.csv', timechart, delimiter=',')

    if save and not singular:
        timechart = timechart.T
        return timechart
    
    #endregion

def shutter(path, save=False, nc_points=200):
    #region init video
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        print("Error: could not open video")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count/fps

    c_history = np.full((frame_count, nc_points, 2), np.nan)

    print(f"FPS: {fps}")
    print(f"frame count: {frame_count}")
    print(f"Resolution: {width} x {height}")
    print(f"duration: {frame_count / fps:.2f}")
    #endregion

    currFrame = 0
    frames = {}
    timechart = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        #region frame handling
        if currFrame not in frames:
            ret, frame = cap.read()
            if not ret: 
                currFrame -= 1
                continue
            frames[currFrame] = frame

        time_seconds = currFrame / fps
        text = f"Frame = {currFrame}, " + f"Time = {time_seconds: .4f}"
        print(text)
        text += f"\nduration: {duration:.2f} \na: back 1 frame\nd: forward 1 frame\nq: close"
        #endregion

        #region frame render
        raw = frames[currFrame]
        result = translate(raw, pointCount=nc_points)
        if result is not None:
            c_points, shown_frame = result
            c_history[currFrame] = c_points
            curr_stable = stability(c_points)
        else:
            shown_frame = raw.copy()

        print(f'stability score: {curr_stable:.2f}\n')
        text += f'\ninstability: {curr_stable:.2f}'
        timechart.append([currFrame,curr_stable])

        cv2.putText(img=shown_frame, text=text, org=(20, 40),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow(path, shown_frame)
        #endregion

        #region frame control
        key = cv2.waitKey(0) & 0xFF
        if key == ord("d"): currFrame += 1
        elif key == ord("a") and currFrame > 0: currFrame -= 1
        elif key == ord("a") and currFrame == 0: continue
        elif key == ord("q"): break
        #endregion

    timechart = np.array(timechart)
    cap.release()
    cv2.destroyAllWindows()

    if save: np.savetxt(path[-12:-4]+'.csv', timechart,delimiter=',')
