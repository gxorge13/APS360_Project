from imports import *

inFolder = 'Data\Cancer&Nodule'
# outFolder = 'Data\ProcessedCancer' # commented out to break code so it doesnt get rerun again

os.makedirs(outFolder, exist_ok=True)

for currImage in os.listdir(inFolder):

    inPath  = os.path.join(inFolder, currImage)
    outPath = os.path.join(outFolder, currImage)

    with Image.open(inPath) as img:
        resized = img.resize((299,299), Image.LANCZOS)
        resized.save(outPath)