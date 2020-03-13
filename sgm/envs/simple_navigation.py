from sgm.dependencies import *

WALLS = {
    'Small':
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
    'Cross':
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
    'FourRooms':
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'Spiral5x5':
        np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 1],
                  [0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 1]]),
    'Spiral7x7':
        np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 1, 0],
                  [1, 0, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 0]]),
    'Spiral9x9':
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 1, 1, 1, 1, 0, 1],
                  [0, 1, 0, 1, 0, 0, 1, 0, 1],
                  [0, 1, 0, 1, 1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1, 0, 1],
                  [0, 1, 1, 1, 1, 1, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]]),
    'Spiral11x11':
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    'Maze3x3':
        np.array([[0, 0, 0],
                  [1, 1, 0],
                  [0, 0, 0]]),
    'Maze6x6':
        np.array([[0, 0, 1, 0, 0, 0],
                  [1, 0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1, 1],
                  [0, 1, 1, 0, 0, 1],
                  [0, 0, 1, 1, 0, 1],
                  [1, 0, 0, 0, 0, 1]]),
    'Maze11x11':
        np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'Tunnel':
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'U':
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]),
    'Tree':
        np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        ]),
    'UMulti':
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    'FlyTrapSmall':
        np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]),
    'FlyTrapBig':
        np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]),
    'Galton':
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
                0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
                0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1,
                0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
                0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]),
}


def resize_walls(walls, factor):
    """Increase the environment by rescaling.

    Args:
      walls: 0/1 array indicating obstacle locations.
      factor: (int) factor by which to rescale the environment."""
    (height, width) = walls.shape
    row_indices = np.array([i for i in range(height) for _ in range(factor)])
    col_indices = np.array([i for i in range(width) for _ in range(factor)])
    walls = walls[row_indices]
    walls = walls[:, col_indices]
    assert walls.shape == (factor * height, factor * width)
    return walls

def thin_wall_width(walls, factor):
    """
    A helper function for thin_walls that thins along the width of walls.
    thin_walls works by calling this method on the walls and walls.T to thin
    along both dimensions.

    Args:
      walls: 0/1 array indicating obstacle locations
      factor: (int) factor by which to rescale the environment
    """
    assert walls.ndim == 2 
    height, width = walls.shape 

    to_the_left = walls
    to_the_right = np.zeros_like(to_the_left)
    to_the_right[:,:-1] = walls[:,1:]
    to_the_right[:,-1] = walls[:,-1]
    filling = to_the_left + to_the_right == 2
    filling = filling.astype(int)
         
    expanded = np.repeat(filling, factor, axis=1)
    expanded[:,0::factor] = walls
    
    assert expanded.shape == (height, width*factor)
    return expanded

def thin_walls(walls, factor):
    """
    Increase the environment by rescaling open space, leaving walls same width.

    Args:
      walls: 0/1 array indicating obstacle locations
      factor: (int) factor by which to rescale the environment
    """
    assert walls.ndim == 2
    height, width = walls.shape
            
    thinned = thin_wall_width(thin_wall_width(walls, factor).T, factor).T
    assert thinned.shape == (height*factor, width*factor)
    return thinned

class PointEnv(gym.Env):
    """Abstract class for 2D navigation environments."""

    def __init__(self, walls=None, resize_factor=1,
                 action_noise=1.0, thin=False):
        """Initialize the point environment.

        Args:
          walls: (str) name of one of the maps defined above.
          resize_factor: (int) Scale the map by this factor.
          action_noise: (float) Standard deviation of noise to add to actions. Use 0
            to add no noise.
        """
        if thin and resize_factor > 1:
            self._walls = thin_walls(WALLS[walls], resize_factor)
        elif not thin and resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]
        self._apsp = self._compute_apsp(self._walls)
        (height, width) = self._walls.shape
        self._height = height
        self._width = width
        self._action_noise = action_noise
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([self._height, self._width]),
            dtype=np.float32)
        self.reset()

    def _sample_empty_state(self):
        candidate_states = np.where(self._walls == 0)
        num_candidate_states = len(candidate_states[0])
        state_index = np.random.choice(num_candidate_states)
        state = np.array([candidate_states[0][state_index],
                          candidate_states[1][state_index]],
                         dtype=np.float)
        state += np.random.uniform(size=2)
        assert not self._is_blocked(state)
        return state

    def reset(self):
        self.state = self._sample_empty_state()
        return self.state.copy()

    def _get_distance(self, obs, goal):
        """Compute the shortest path distance.

        Note: This distance is *not* used for training."""
        (i1, j1) = self._discretize_state(obs)
        (i2, j2) = self._discretize_state(goal)
        return self._apsp[i1, j1, i2, j2]

    def _discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int)
        # Round down to the nearest cell if at the boundary.
        if i == self._height:
            i -= 1
        if j == self._width:
            j -= 1
        return (i, j)

    def _is_blocked(self, state):
        if not self.observation_space.contains(state):
            return True
        (i, j) = self._discretize_state(state)
        return (self._walls[i, j] == 1)

    def step(self, action):
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                if not self._is_blocked(new_state):
                    self.state = new_state

        done = False
        rew = -1.0 * np.linalg.norm(self.state)
        return self.state.copy(), rew, done, {}

    @property
    def walls(self):
        return self._walls

    def _compute_apsp(self, walls):
        (height, width) = walls.shape
        g = nx.Graph()
        # Add all the nodes
        for i in range(height):
            for j in range(width):
                if walls[i, j] == 0:
                    g.add_node((i, j))

        # Add all the edges
        for i in range(height):
            for j in range(width):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == dj == 0:
                            continue  # Don't add self loops
                        if i + di < 0 or i + di > height - 1:
                            continue  # No cell here
                        if j + dj < 0 or j + dj > width - 1:
                            continue  # No cell here
                        if walls[i, j] == 1:
                            continue  # Don't add edges to walls
                        if walls[i + di, j + dj] == 1:
                            continue  # Don't add edges to walls
                        g.add_edge((i, j), (i + di, j + dj))

        # dist[i, j, k, l] is path from (i, j) -> (k, l)
        dist = np.full((height, width, height, width), np.float('inf'))
        for ((i1, j1), dist_dict) in nx.shortest_path_length(g):
            for ((i2, j2), d) in dist_dict.items():
                dist[i1, j1, i2, j2] = d
        return dist


class GoalConditionedPointWrapper(gym.Wrapper):
    """Wrapper that appends goal to observation produced by environment."""

    def __init__(self, env, prob_constraint=0.8, min_dist=0, max_dist=4,
                 threshold_distance=1.0):
        """Initialize the environment.

        Args:
          env: an environment.
          prob_constraint: (float) Probability that the distance constraint is
            followed after resetting.
          min_dist: (float) When the constraint is enforced, ensure the goal is at
            least this far from the initial observation.
          max_dist: (float) When the constraint is enforced, ensure the goal is at
            most this far from the initial observation.
          threshold_distance: (float) States are considered equivalent if they are
            at most this far away from one another.
        """
        self._threshold_distance = threshold_distance
        self._prob_constraint = prob_constraint
        self._min_dist = min_dist
        self._max_dist = max_dist
        super(GoalConditionedPointWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Dict({
            'observation': env.observation_space,
            'goal': env.observation_space,
        })

    def _normalize_obs(self, obs):
        return np.array([
            obs[0] / float(self.env._height),
            obs[1] / float(self.env._width)
        ])

    def reset(self):
        goal = None
        count = 0
        while goal is None:
            obs = self.env.reset()
            (obs, goal) = self._sample_goal(obs)
            count += 1
            if count > 1000:
                print('WARNING: Unable to find goal within constraints.')
        self._goal = goal
        return {'observation': self._normalize_obs(obs),
                'goal': self._normalize_obs(self._goal)}

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        rew = -1.0
        done = self._is_done(obs, self._goal)
        return {'observation': self._normalize_obs(obs),
                'goal': self._normalize_obs(self._goal)}, rew, done, {}

    def set_sample_goal_args(self, prob_constraint=None,
                             min_dist=None, max_dist=None):
        assert prob_constraint is not None
        assert min_dist is not None
        assert max_dist is not None
        assert min_dist >= 0
        assert max_dist >= min_dist
        self._prob_constraint = prob_constraint
        self._min_dist = min_dist
        self._max_dist = max_dist

    def _is_done(self, obs, goal):
        """Determines whether observation equals goal."""
        return np.linalg.norm(obs - goal) < self._threshold_distance

    def _sample_goal(self, obs):
        """Sampled a goal observation."""
        if np.random.random() < self._prob_constraint:
            return self._sample_goal_constrained(obs, self._min_dist, self._max_dist)
        else:
            return self._sample_goal_unconstrained(obs)

    def _sample_goal_constrained(self, obs, min_dist, max_dist):
        """Samples a goal with dist min_dist <= d(observation, goal) <= max_dist.

        Args:
          obs: observation (without goal).
          min_dist: (int) minimum distance to goal.
          max_dist: (int) maximum distance to goal.
        Returns:
          observation: observation (without goal).
          goal: a goal observation.
        """
        (i, j) = self.env._discretize_state(obs)
        mask = np.logical_and(self.env._apsp[i, j] >= min_dist,
                              self.env._apsp[i, j] <= max_dist)
        mask = np.logical_and(mask, self.env._walls == 0)
        candidate_states = np.where(mask)
        num_candidate_states = len(candidate_states[0])
        if num_candidate_states == 0:
            return (obs, None)
        goal_index = np.random.choice(num_candidate_states)
        goal = np.array([candidate_states[0][goal_index],
                         candidate_states[1][goal_index]],
                        dtype=np.float)
        goal += np.random.uniform(size=2)
        dist_to_goal = self.env._get_distance(obs, goal)
        assert min_dist <= dist_to_goal <= max_dist
        assert not self.env._is_blocked(goal)
        return (obs, goal)

    def _sample_goal_unconstrained(self, obs):
        """Samples a goal without any constraints.

        Args:
          obs: observation (without goal).
        Returns:
          observation: observation (without goal).
          goal: a goal observation.
        """
        return (obs, self.env._sample_empty_state())

    @property
    def max_goal_dist(self):
        apsp = self.env._apsp
        return np.max(apsp[np.isfinite(apsp)])


class NonTerminatingTimeLimit(wrappers.PyEnvironmentBaseWrapper):
    """Resets the environment without setting done = True.

    Resets the environment if either these conditions holds:
      1. The base environment returns done = True
      2. The time limit is exceeded.
    """

    def __init__(self, env, duration):
        super(NonTerminatingTimeLimit, self).__init__(env)
        self._duration = duration
        self._step_count = None

    def _reset(self):
        self._step_count = 0
        return self._env.reset()

    @property
    def duration(self):
        return self._duration

    def _step(self, action):
        if self._step_count is None:
            return self.reset()

        ts = self._env.step(action)

        self._step_count += 1
        if self._step_count >= self._duration or ts.is_last():
            self._step_count = None

        return ts


def env_load_fn(environment_name,
                max_episode_steps=None,
                resize_factor=1,
                gym_env_wrappers=(GoalConditionedPointWrapper,),
                terminate_on_timeout=False,
                thin=False):
    """Loads the selected environment and wraps it with the specified wrappers.

    Args:
      environment_name: Name for the environment to load.
      max_episode_steps: If None the max_episode_steps will be set to the default
        step limit defined in the environment's spec. No limit is applied if set
        to 0 or if there is no timestep_limit set in the environment's spec.
      gym_env_wrappers: Iterable with references to wrapper classes to use
        directly on the gym environment.
      terminate_on_timeout: Whether to set done = True when the max episode
        steps is reached.

    Returns:
      A PyEnvironmentBase instance.
    """
    gym_env = PointEnv(walls=environment_name,
                       resize_factor=resize_factor,
                       thin=thin)

    for wrapper in gym_env_wrappers:
        gym_env = wrapper(gym_env)
    env = gym_wrapper.GymWrapper(
        gym_env,
        discount=1.0,
        auto_reset=True,
    )

    if max_episode_steps > 0:
        if terminate_on_timeout:
            env = wrappers.TimeLimit(env, max_episode_steps)
        else:
            env = NonTerminatingTimeLimit(env, max_episode_steps)

    return tf_py_environment.TFPyEnvironment(env)


def plot_walls(walls):
    walls = walls.T
    (height, width) = walls.shape
    for (i, j) in zip(*np.where(walls)):
        x = np.array([j, j+1]) / float(width)
        y0 = np.array([i, i]) / float(height)
        y1 = np.array([i+1, i+1]) / float(height)
        plt.fill_between(x, y0, y1, color='grey')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks([])
    plt.yticks([])


if __name__ == "__main__":

    plt.figure(figsize=(12, 7))
    for index, (name, walls) in enumerate(WALLS.items()):
        plt.subplot(3, 6, index + 1)
        plt.title(name)
        plot_walls(walls)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.suptitle('Navigation Environments', fontsize=20)
    plt.show()
