import pydub
import os

def export_wav (file_name):
        file_path = file_name
        music = pydub.AudioSegment.from_file(file_path)
        duration = music.duration_seconds
        middel_of_segment = int (duration / 2)
        start_time = middel_of_segment 
        end_time = start_time + 30
        segment = music[start_time * 1000 : end_time * 1000]
        segment.export(f"D:\\MyLesson\\کارشناسی\\ترم هفت\\مبانی یادگیری ماشین\\Project\\Source\\Demo\\{file_name[0:len(file_name) - 4:1]}_Demo.wav".format(),format = 'wav')
def Creat_demo():
    Direct_Path = "D:\MyLesson\کارشناسی\ترم هفت\مبانی یادگیری ماشین\Project\Source"
    os.chdir("Source")
    os.chdir("Main_File")
    Music_files = os.listdir()
    
    for file in Music_files:
        export_wav(file)
