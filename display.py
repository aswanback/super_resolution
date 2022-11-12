import matplotlib.pyplot as plt

def plot_results(lr,sr,hr):
  fig, ax = plt.subplots(3,1)
  fig.set_size_inches(48,48)
  ax[0].imshow(lr.permute(1,2,0))
  ax[1].imshow(sr.permute(1,2,0))
  ax[2].imshow(hr.permute(1,2,0))
  fig.show()
