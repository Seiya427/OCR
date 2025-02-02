import torch
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import ocr
import matplotlib.pyplot as plt
from textblob import TextBlob
from spellchecker import SpellChecker
import cv2

network = ocr.CNN2()
saveFile = torch.load(ocr.path2)
network.load_state_dict(saveFile['state_dict'])


def preprocessImage(image):
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.point(lambda x: 0 if x < 32 else 255, 'L')
    image = np.array(image)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = Image.fromarray(image,'L')
    image = image.filter(ImageFilter.GaussianBlur(radius=0.2))
    binaryImage = image.point(lambda x: 0 if x < 100 else 255, '1')
    return binaryImage

def findLines(image):
    imageData = np.array(image)
    horizontalProjection = np.sum(imageData, axis=1)
    
    lines = []
    lineStart = None
    
    for i, value in enumerate(horizontalProjection):
        if value > 0 and lineStart is None:
            lineStart = i
        elif value == 0 and lineStart is not None:
            lines.append(image.crop((0, lineStart, image.width, i)))
            lineStart = None
    
    if lineStart is not None:
        lines.append(image.crop((0, lineStart, image.width, image.height)))
    
    return lines

def findWords(lineImage, wordGap=5):
    lineData = np.array(lineImage)
    verticalProjection = np.sum(lineData, axis=0)
    
    words = []
    wordStart = None
    gapSize = 0
    
    for i, value in enumerate(verticalProjection):
        if value > 0: 
            if wordStart is None:
                wordStart = i
            gapSize = 0 
        elif value == 0 and wordStart is not None:  
            gapSize += 1
            if gapSize >= wordGap: 
                wordEnd = i - gapSize
                words.append(lineImage.crop((wordStart, 0, wordEnd, lineImage.height)))
                wordStart = None 
    
    if wordStart is not None:
        words.append(lineImage.crop((wordStart, 0, lineImage.width, lineImage.height)))
    
    return words

def mergeBoundingBoxes(boundingBoxes):
    boundingBoxes = sorted(boundingBoxes, key=lambda box: box[0])
    mergedBoxes = []
    while boundingBoxes:
        x, y, w, h = boundingBoxes.pop(0)
        mergedBox = [x, y, x + w, y + h]
        
        indicesToRemove = []
        for i, (x2, y2, w2, h2) in enumerate(boundingBoxes):
            if x2 >= x and x2+w2 <= x+w:
                mergedBox[0] = min(mergedBox[0], x2)
                mergedBox[1] = min(mergedBox[1], y2)
                mergedBox[2] = max(mergedBox[2], x2 + w2)
                mergedBox[3] = max(mergedBox[3], y2 + h2)
                indicesToRemove.append(i)
        for index in sorted(indicesToRemove, reverse=True):
            del boundingBoxes[index]
        mergedBoxes.append((mergedBox[0], mergedBox[1], mergedBox[2] - mergedBox[0], mergedBox[3] - mergedBox[1]))
    
    return mergedBoxes

def findCharacters(wordImage, debug=False):
    openCvImage = np.array(wordImage, dtype=np.uint8) * 255 
    originalImage = np.array(wordImage.convert('RGB'))
    _, thresh = cv2.threshold(openCvImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
    boundingBoxes = sorted(boundingBoxes, key=lambda box: box[0]) 
    boundingBoxes = mergeBoundingBoxes(boundingBoxes)
    for x, y, w, h in boundingBoxes:
        if debug:
            cv2.rectangle(originalImage, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    if debug:
        displayImage(originalImage)

    characters = []
    for x, y, w, h in boundingBoxes:
        character = wordImage.crop((x, y, x + w, y + h))
        characters.append(character)
    
    characters = sorted(characters, key=lambda char: char.getbbox()[0], reverse = True)
    
    return characters

def cropImage(image):
    imageData = np.array(image)
    verticalProjection = np.sum(imageData, axis=0)
    horizontalProjection = np.sum(imageData, axis=1)
    top = next((i for i, val in enumerate(horizontalProjection) if val > 0), 0)
    bottom = next((i for i, val in enumerate(reversed(horizontalProjection)) if val > 0), 0)
    bottom = len(horizontalProjection) - bottom  
    
    left = next((i for i, val in enumerate(verticalProjection) if val > 0), 0)
    right = next((i for i, val in enumerate(reversed(verticalProjection)) if val > 0), 0)
    right = len(verticalProjection) - right 

    croppedImage = image.crop((left, top, right, bottom))
    return croppedImage

def resizeAndPad(characterImage):
    
    height = characterImage.height
    width = characterImage.width
    scaleFactor = 26 / height
    newWidth = int(width * scaleFactor)
    characterImage = characterImage.resize((newWidth, 26), Image.LANCZOS)
    newImage = Image.new('L', (28, 28), color=0)
    newImage.paste(characterImage, (14 - newWidth // 2, 1))
    return newImage

def characterToTensor(characterImage):
    characterArray = np.array(characterImage) / 255.0
    tensor = torch.tensor(characterArray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def displayImage(image, grayscale = True):
    if grayscale:
        plt.imshow(image, cmap = 'gray')
    else:
        plt.imshow(image)
    plt.show()
    
