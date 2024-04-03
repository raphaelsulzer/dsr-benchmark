from PIL import Image, ImageDraw
from glob import glob
import os
import numpy as np

def png2jpg(infile,outfile):

    # Load your two images
    im = Image.open(infile)

    bg = Image.new("RGB", im.size, "WHITE")

    bg.paste(im, (0, 0), im)
    # Save the resulting image

    # bg = bg.resize((int(bg.size[0]/2),int(bg.size[1]/2)))

    bg.save(outfile, subsampling=0, quality=90, optimize=True)


if __name__ == "__main__":

    inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/defects/"
    outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/defects_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/survey/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/survey_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/ignatius/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/ignatius_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/shapenet3000/02691156/d18592d9615b01bbbc0909d98a1ff2b4/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/shapenet3000/02691156/d18592d9615b01bbbc0909d98a1ff2b4_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/shapenet/02691156/d18592d9615b01bbbc0909d98a1ff2b4/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/shapenet/02691156/d18592d9615b01bbbc0909d98a1ff2b4_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/modelnet/bed/0585/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/modelnet/bed/0585_jpg/"
    #
    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/reconbench/daratech/daratech/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/reconbench/daratech/daratech_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/real/templeRing/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/real/templeRing_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/real/scan1/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/real/scan1_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/real/Truck/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/real/Truck_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/exp/02691156/1bea1445065705eb37abdc1aa610476c/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/exp/02691156/1bea1445065705eb37abdc1aa610476c_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/exp/dc/dc/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/exp/dc/dc_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/exp/table/0008/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/exp/table/0008_jpg/"

    # inpath = "/home/rsulzer/data/SurveyAndBenchmark/images/exp/table/0470/"
    # outpath = "/home/rsulzer/data/SurveyAndBenchmark/images/exp/table/0470_jpg/"

    os.makedirs(outpath,exist_ok=True)
    for file in os.listdir(inpath):

        if not file[-4:] == ".png": continue

        if not os.path.isfile(os.path.join(inpath,file)):
            continue

        infile = os.path.join(inpath,file)
        outfile = os.path.join(outpath,file[:-4]+".jpg")

        png2jpg(infile,outfile)





