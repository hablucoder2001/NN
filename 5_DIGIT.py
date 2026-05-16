import numpy as np

digits = {
    1: np.array([
        [0,0,1,0,0],
        [0,1,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,1,1,1,0]
    ]),

    2: np.array([
        [1,1,1,1,0],
        [0,0,0,1,0],
        [1,1,1,1,0],
        [1,0,0,0,0],
        [1,1,1,1,1]
    ]),

    3: np.array([
        [1,1,1,1,0],
        [0,0,0,1,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
        [1,1,1,1,0]
    ]),

    4: np.array([
        [1,0,0,1,0],
        [1,0,0,1,0],
        [1,1,1,1,0],
        [0,0,0,1,0],
        [0,0,0,1,0]
    ]),

    5: np.array([
        [1,1,1,1,1],
        [1,0,0,0,0],
        [1,1,1,1,0],
        [0,0,0,1,0],
        [1,1,1,1,0]
    ])
}


def recognize_image(image):
    image_vec = image.flatten()
    best_digit = None
    min_error = float('inf')

    for digit, template in digits.items():
        template_vec = template.flatten()

        err = np.sum(image_vec != template_vec)

        if err < min_error:
            min_error = err
            best_digit = digit

    return best_digit



test_image = np.array([
    [1,1,1,1,0],
    [0,0,1,0,0],
    [0,1,1,0,0],
    [1,0,0,0,0],
    [0,1,1,1,0]
])


print(recognize_image(test_image))
