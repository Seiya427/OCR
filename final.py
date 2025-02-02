import torch
import ocr
from segmentation import *
from textblob import TextBlob
import matplotlib.pyplot as plt

def postprocessText(word, errorMargin = 5):
    word = word.lower()
    word = list(word)
    for i, letter in enumerate(word):
        match letter:
            case '1': word[i] = 'i'
            case '2': word[i] = 'a'
            case '4': word[i] = 'y'
            case '5': word[i] = 's'
            case '9': word[i] = 'q'
            case '0': word[i] = 'o'
    word = ''.join(word)
    word = str(TextBlob(word).correct())
    return word

def prettyPrint(textArray):
    rows = [' '.join(row) for row in textArray]
    for row in rows:
        print(row)

def recognizeCharacters(imagePath):
    image = Image.open(imagePath)
    binaryImage = preprocessImage(image)
    resultText = []
    resultLine = []
    resultWord = ''
    lines = findLines(binaryImage)
    for line in lines:
        words = findWords(line)
        for word in words:
            characters = findCharacters(word)
            for character in characters:
                characterImage = resizeAndPad(cropImage(character))
                characterTensor = characterToTensor(characterImage)
                with torch.no_grad():
                    output = network(characterTensor)
                    _, predictedCharacter = torch.max(output, 1)
                resultWord += ocr.classes[predictedCharacter]
            resultWord = postprocessText(resultWord)
            resultLine.append(resultWord)
            resultWord = ''
        resultText.append(resultLine)
        resultLine = []
    return resultText

if __name__ == '__main__':
    recognizeCharacters('input_image.png')