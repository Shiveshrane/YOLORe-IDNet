class camera_Recommender:
    def __init__(self, distances=None, window_size=2):
        self.last_camera_id = None
        self.last_index = -(window_size + 1)
        # self.distances = distances or [[0, 1, 900, 999],
        #                                [1, 0, 900, 999],
        #                                [900, 999, 0, 1],
        #                                [900, 999, 1, 0]]

        self.distances = distances or [
                            [0, 1, 2, 3, 4, 5, 6, 7, 8],
                            [1, 0, 1, 2, 3, 4, 5, 6, 7],
                            [2, 1, 0, 1, 2, 3, 4, 5, 6],
                            [3, 2, 1, 0, 1, 2, 3, 4, 5],
                            [4, 3, 2, 1, 0, 1, 2, 3, 4],
                            [5, 4, 3, 2, 1, 0, 1, 2, 3],
                            [6, 5, 4, 3, 2, 1, 0, 1, 2],
                            [7, 6, 5, 4, 3, 2, 1, 0, 1],
                            [8, 7, 6, 5, 4, 3, 2, 1, 0]
                            ]
        self.window_size = window_size

    def get_nearest_cameras(self, camera_id):
        if self.last_camera_id != camera_id:
            self.last_camera_id = camera_id
            self.last_index = -(self.window_size + 1) # if the camera has changed, we need to start looking for the closest cameras again from the beginning

        distances_for_camera = self.distances[camera_id]
        num_cameras = len(distances_for_camera)
        closest_cameras = []

        if self.last_index == -(self.window_size + 1):
            closest_cameras = sorted(range(num_cameras), key=lambda i: distances_for_camera[i])[0:self.window_size]
            self.last_index = 0

        elif self.last_index < num_cameras - self.window_size:
            closest_cameras = sorted(range(num_cameras), key=lambda i: distances_for_camera[i])[self.last_index+self.window_size:self.last_index+(2*self.window_size)]
            self.last_index += self.window_size

        else:
            closest_cameras = sorted(range(num_cameras), key=lambda i: distances_for_camera[i])[0:self.window_size]
            self.last_index = 0
        return closest_cameras
