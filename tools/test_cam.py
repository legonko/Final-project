import pyrealsense2 as rs
import numpy as np
import cv2
import time



def merge_frames(frame_front, frame_back):
    '''merge frames from front and back cameras'''
    frame_back = frame_back[::-1] # vertical mirror
    frame_back = frame_back[:, ::-1] # horizontal mirror
    merged_frame = np.concatenate((frame_front, frame_back), axis=0)
    return merged_frame

def ipm_ll(image, homography_matrix):
    image = np.asanyarray(image, dtype=np.uint8)
    image = cv2.resize(image, dsize=(640, 480))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transformed_image = cv2.warpPerspective(image, homography_matrix, (640, 480))

    return transformed_image

def find_homography():
    
    # for 13 cm height
    x1, y1 = 290, 480
    x2, y2 = 300, 380
    x3, y3 = 360, 380
    x4, y4 = 370, 480

    u1, v1 = x1, y1
    u2, v2 = 275, 0
    u3, v3 = 370, 0
    u4, v4 = x4, y4
    
    # camera points
    points_camera = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
    # birds-eye view points
    points_birdseye = np.array([[u1, v1], 
                                [u2, v2], 
                                [u3, v3], 
                                [u4, v4]], dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(points_camera, points_birdseye)
    
    return homography_matrix

def rs_stream_2(H):
    realsense_ctx = rs.context()
    connected_devices = []
    for i in range(len(realsense_ctx.devices)):
        detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
        connected_devices.append(detected_camera)
    
    pipe1 = rs.pipeline()
    cnfg1  = rs.config()
    cnfg1.enable_device(connected_devices[0])
    pipe2 = rs.pipeline()
    cnfg2  = rs.config()
    cnfg2.enable_device(connected_devices[1])

    cnfg1.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
    cnfg1.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
    cnfg2.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
    cnfg2.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

    profile1 = pipe1.start(cnfg1)
    profile2 = pipe2.start(cnfg2)

    align1 = rs.align(rs.stream.color)
    align2 = rs.align(rs.stream.color)

    frames1 = pipe1.wait_for_frames()
    frames2 = pipe2.wait_for_frames()
    time.sleep(5)
    aligned_frames1 = align1.process(frames1)
    color_frame1 = aligned_frames1.get_color_frame()
    aligned_frames2 = align2.process(frames2)
    color_frame2 = aligned_frames2.get_color_frame()
    t0 = time.time()
    color_image1 = np.asanyarray(color_frame1.get_data())
    color_image2 = np.asanyarray(color_frame2.get_data())


    while True:
        start_time = time.time()
        frames1 = pipe1.wait_for_frames()
        frames2 = pipe2.wait_for_frames()

        aligned_frames1 = align1.process(frames1)
        depth_frame1 = aligned_frames1.get_depth_frame()
        color_frame1 = aligned_frames1.get_color_frame()
        aligned_frames2 = align2.process(frames2)
        depth_frame2 = aligned_frames2.get_depth_frame()
        color_frame2 = aligned_frames2.get_color_frame()
        dt = time.time() - t0 #if t1 != None else None
        t0 = time.time()

        if not color_frame1 or not depth_frame1 or not color_frame2 or not depth_frame2:
            continue

        depth_image1 = np.asanyarray(depth_frame1.get_data())
        color_image1 = np.asanyarray(color_frame1.get_data())
        depth_image2 = np.asanyarray(depth_frame2.get_data())
        color_image2 = np.asanyarray(color_frame2.get_data())
        color_image1 = ipm_ll(color_image1, H)
        color_image2 = ipm_ll(color_image2, H)
        merged_map = merge_frames(color_image2, color_image1)
    
        cv2.imshow('merged map', merged_map)

        if cv2.waitKey(1) == ord('q'):
            break

        end_time = time.time()

        print('loop time: ', round(end_time-start_time, 4))
    
    pipe1.stop()
    pipe2.stop()
    cv2.destroyAllWindows() 

if __name__ == '__main__':


    H = find_homography()
    rs_stream_2(H)
    # cv_stream(model)
    # img = Image.open('cv_frame.jpg').convert("RGB")  # rs_color_img2.jpg
    # put_img(model, img)
    cv2.destroyAllWindows()