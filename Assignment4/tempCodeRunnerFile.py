fig, ax = plt.subplots()
ax.imshow(magnitude_spectrum_normalized, cmap='gray', extent=[-img.shape[1]//2, img.shape[1]//2, -img.shape[0]//2, img.shape[0]//2])
ax.set_xlabel("Frequency (u)")
ax.set_ylabel("Frequency (v)")
cid = fig.canvas.mpl_connect('button_press_event', onclick)