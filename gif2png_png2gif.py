import imageio

def gif2png(gifPath, imagePrefix="image "):
    reader = imageio.get_reader(gifPath)
    for i,im in enumerate(reader):
        imageio.imwrite(imagePrefix+str(i)+".png", im)

def png2gif(pngList, gifPath):
    images=[]
    for png in pngList:
        images.append(imageio.imread(png))
    imageio.mimsave(gifPath, images)
