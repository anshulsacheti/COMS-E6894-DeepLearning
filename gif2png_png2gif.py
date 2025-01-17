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

#Credit to https://gist.github.com/BigglesZX/4016539

def analyseImage(path):
    '''
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    '''
    im = Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results

def processImage(path):
    '''
    Iterate the GIF, extracting each frame.
    '''
    mode = analyseImage(path)['mode']
    im = Image.open(path)
    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')
    try:
        while True:
            print "saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile)
            '''
            If the GIF uses local colour tables, each frame will have its own palette.
            If not, we need to apply the global palette to the new frame.
            '''
            if not im.getpalette():
                im.putpalette(p)
            new_frame = Image.new('RGBA', im.size)
            '''
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
            If so, we need to construct the new frame by pasting it on top of the preceding frames.
            '''
            if mode == 'partial':
                new_frame.paste(last_frame)
            new_frame.paste(im, (0,0), im.convert('RGBA'))
            new_frame.save('%s-%d.png' % (''.join(os.path.basename(path).split('.')[:-1]), i), 'PNG')
            i += 1
            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass
