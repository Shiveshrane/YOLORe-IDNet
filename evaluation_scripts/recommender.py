# distances = [[0, 1, 999, 999, 1, 1, 999, 1, 1],
#                     [1, 0, 999, 999, 1, 1, 999, 1, 1],
#                     [999, 999, 0, 1, 999, 999, 1, 1, 999],
#                     [999, 999, 1, 0, 999, 999, 1, 1, 999],
#                     [1, 1, 999, 999, 0, 1, 999, 1, 1],
#                     [1, 1, 999, 999, 1, 0, 999, 1, 1],
#                     [999, 999, 1, 1, 999, 999, 0, 1, 999],
#                     [1, 1, 1, 1, 1, 1, 1, 0, 999],
#                     [1, 1, 999, 999, 1, 1, 999, 999, 0]]

class camera_Recommender:
    def __init__(self, distances=None, window_size=1):
        self.last_camera_id = None
        self.last_index = -(window_size + 1)
        # self.distances = distances or [[0, 1, 900, 999],
        #                                [1, 0, 900, 999],
        #                                [900, 999, 0, 1],
        #                                [900, 999, 1, 0]]

        self.distances = distances or [
                            [0]
                            ]
        self.window_size = window_size

    def get_nearest_cameras(self, camera_id):
        if self.last_camera_id != camera_id:
            # If the new camera_id is different than the previous one, start over
            self.last_camera_id = camera_id
            self.last_index = -(self.window_size + 1)

        distances_for_camera = self.distances[camera_id]
        num_cameras = len(distances_for_camera)
        closest_cameras = []

        if self.last_index == -(self.window_size + 1):
            # If no closest cameras have been returned yet, find the first window_size closest cameras
            closest_cameras = sorted(range(num_cameras), key=lambda i: distances_for_camera[i])[0:self.window_size]
            self.last_index = 0

        elif self.last_index < num_cameras - self.window_size:
            # If there are more closest cameras to return, slide the window by window_size and return the next window_size closest cameras
            closest_cameras = sorted(range(num_cameras), key=lambda i: distances_for_camera[i])[self.last_index+self.window_size:self.last_index+(2*self.window_size)]
            self.last_index += self.window_size

        else:
            # If there are no more closest cameras to return, start over from the beginning
            closest_cameras = sorted(range(num_cameras), key=lambda i: distances_for_camera[i])[0:self.window_size]
            self.last_index = 0

        return closest_cameras
