from timeit import default_timer as timer


class FpsCalculator():
    def __init__(self) -> None:

        self.frame_count = 0

        self.accum_time = 0
        self.curr_fps = 0
        self.prev_time = timer()
        self.result_fps = 0

    def calc(self):
        # update frame count
        self.frame_count += 1

        # update fps
        self.__curr_time = timer()
        self.__exec_time = self.__curr_time - self.prev_time
        self.prev_time = self.__curr_time
        self.accum_time = self.accum_time + self.__exec_time
        self.curr_fps = self.curr_fps + 1
        if self.accum_time > 1:
            self.accum_time = self.accum_time - 1
            self.result_fps = self.curr_fps
            self.curr_fps = 0

        return self.result_fps
