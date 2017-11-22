import glob
from moviepy.editor import ImageSequenceClip

images_dir = './results/baby30/train_latest/videos/7/'
## Change XX to be objective images name
v = 'g05_c02'
images_list = glob.glob(images_dir + "v_BabyCrawling_{}**.png".format(v))
images_list.sort()
ouput_file = './{}.mp4'.format(v)
fps = 1



if __name__ == '__main__':
    clip = ImageSequenceClip(images_list, fps=fps)

    clip.write_videofile(ouput_file, fps=fps, audio=False)
