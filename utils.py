from timeit import default_timer as timer
import cv2


class DispFps():
    def __init__(self):
        # 表示関連定義
        self.__width = 80
        self.__height = 20
        self.__font_size = 0.4
        self.__font_width = 1
        self.__font_style = cv2.FONT_HERSHEY_COMPLEX
        self.__font_color = (255, 0, 0)
        self.__background_color = (0, 0, 0)

        # フレーム数カウント用変数
        self.__frame_count = 0

        # FPS計算用変数
        self.__accum_time = 0
        self.__curr_fps = 0
        self.__prev_time = timer()
        self.__str = "FPS: "

    def __calc(self):
        # フレーム数更新
        self.__frame_count += 1

        # FPS更新
        self.__curr_time = timer()
        self.__exec_time = self.__curr_time - self.__prev_time
        self.__prev_time = self.__curr_time
        self.__accum_time = self.__accum_time + self.__exec_time
        self.__curr_fps = self.__curr_fps + 1
        if self.__accum_time > 1:
            self.__accum_time = self.__accum_time - 1
            self.__str = "FPS: " + str(self.__curr_fps)
            self.__curr_fps = 0

    def __disp(self, frame, str, x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.__background_color, -1)
        cv2.putText(frame, str, (x1 + 5, y2 - 5), self.__font_style,
                    self.__font_size, self.__font_color, self.__font_width)

    def disp(self, frame):
        # 表示内容計算
        self.__calc()
        # フレーム数（左上に表示する）
        self.__disp(frame, str(self.__frame_count), 0,
                    0, x2=self.__width, y2=self.__height)
        # FPS(右上に表示する)
        screen_width = int(frame.shape[1])
        self.__disp(frame, self.__str, screen_width -
                    self.__width, 0, screen_width, self.__height)
