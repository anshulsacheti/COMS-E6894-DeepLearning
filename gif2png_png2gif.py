import imageio

def gif2png(gifPath, imagePrefix="image "):
    """Uses imageio to convert a set of gif into png files
    Inputs:
    gifPath: Path of generated gif file
    imagePrefix: fileName that all generated pngs should start with. png all
                 generated in same directory as gif
    """
    reader = imageio.get_reader(gifPath)
    for i,im in enumerate(reader):
        imageio.imwrite(imagePrefix+str(i)+".png", im)

def png2gif(pngList, gifPath):
    """Uses imageio to convert a set of png images into a gif
    Inputs:
    pngList: array of png filename strings
    gifPath: Path of generated gif file
    """
    images=[]
    for png in pngList:
        images.append(imageio.imread(png))
    imageio.mimsave(gifPath, images)
