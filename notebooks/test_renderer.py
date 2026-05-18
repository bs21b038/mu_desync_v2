import matplotlib.pyplot as plt

from src.data.renderer import TensorRenderer

renderer = TensorRenderer()

coords = [-0.15,0, 0.15,0]
target = [0.5, 0.5]

img = renderer.render(coords, target)

img_np = img.permute(1, 2, 0).numpy()

plt.imshow(img_np)
plt.axis("off")
plt.show()