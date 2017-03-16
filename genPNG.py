import imageio

reader = imageio.get_reader("movie.gif")
for i,im in enumerate(reader):
    imageio.imwrite("image "+str(i)+".png", im)
