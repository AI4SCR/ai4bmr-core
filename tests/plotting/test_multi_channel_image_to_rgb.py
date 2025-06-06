
def test_multi_channel_image_to_rgb():
    import numpy as np
    from ai4bmr_core.plotting.utils import image_to_rgba
    import matplotlib.pyplot as plt
    # TODO: we need the image in shape (C, H, W) for the plotting function but we get back (H, W, 3) which is not intuitive
    image = np.random.rand(5, 10, 10)
    rgb_image = image_to_rgba(image=image)
    plt.imshow(rgb_image).figure.show()